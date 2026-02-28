
 **where data is stored, how it flows, exact locations, table names, file formats, and folder structures**

---

# 📌 1️⃣ Overall Storage Architecture

The project is built inside:

```
Catalog: workspace
Schema: default + legal_data
Storage Layer: Unity Catalog Volumes
Data Format: Delta Lake (Parquet-based)
Vector Store: Chroma DB + Delta Embedding Storage
Owner: kumarangad54706@gmail.com
```

You are using a **Medallion Architecture**:

```
Bronze → Silver → Gold → Vector Storage
```

---

# 📂 2️⃣ Physical Storage Locations (Unity Catalog Volumes)

All physical files are stored under:

```
/Volumes/workspace/legal_data/
```

Inside this root path, you have:

```
/Volumes/workspace/legal_data/
│
├── bronze/
├── silver/
├── gold/
├── chroma_db/
├── vector_db/
├── vector_db_test/
├── raw_documents/
```

---

# 🥉 3️⃣ Bronze Layer – Raw Legal Documents

### 🔹 Table Name:

```
workspace.default.bronze_legal_documents
```

### 🔹 Table Type:

* Managed
* Delta format

### 🔹 Physical Storage Location:

```
/Volumes/workspace/legal_data/bronze/legal_documents/
```

### 🔹 File Format:

```
Snappy Compressed Parquet (Delta)
```

Example files:

```
part-00000-6997b044-94c1-45df-9725-6585adf5f335.c000.snappy.parquet
part-00000-cc8ae997-923c-4900-b725-128ccfc5a389.c000.snappy.parquet
part-00000-d70013ab-8aeb-4106-a6b1-c02d810c7600.c000.snappy.parquet
```

### 🔹 Columns Stored:

* doc_id
* file_name
* category
* file_path
* raw_text
* page_count
* file_size_kb
* file_hash
* modified_time
* is_scanned
* ingestion_time
* status

### 🔹 Purpose:

Stores raw extracted text from legal PDFs before section-level parsing.

---

# 🥈 4️⃣ Silver Layer – Section-Level Structured Data

### 🔹 Table Name:

```
workspace.default.silver_legal_sections
```

### 🔹 Table Type:

* Managed
* Delta format

### 🔹 Physical Storage Location:

```
/Volumes/workspace/legal_data/silver/legal_sections/
```

### 🔹 File Format:

Snappy compressed parquet (Delta)

Example files:

```
part-00000-b6d01f2b-aabd-4eb3-9f50-cfc5823a2f81.c000.snappy.parquet
part-00001-93437962-d536-42e7-b548-93cfc1c2e609.c000.snappy.parquet
```

### 🔹 Columns Stored:

* section_id
* doc_id
* file_name
* category
* act_name
* section_number
* section_title
* section_text
* source_path
* created_at

### 🔹 Purpose:

Transforms raw legal text into structured legal sections.

This is your **legal semantic unit layer**.

---

# 🥇 5️⃣ Gold Layer – Chunked Legal Text for Embeddings

### 🔹 Table Name:

```
workspace.default.gold_legal_chunks
```

### 🔹 Table Type:

* Managed
* Delta format

### 🔹 Physical Storage Location:

```
/Volumes/workspace/legal_data/gold/legal_chunks/
```

### 🔹 File Format:

Snappy compressed parquet (Delta)

Example files:

```
part-00000-a12cbee6-14fa-41a2-8f82-fc77186e4d2f.c000.snappy.parquet
part-00001-57542e29-10e3-487f-b0cf-18ff05b9ab55.c000.snappy.parquet
...
```

### 🔹 Columns Stored:

* act_name
* category
* char_count
* chunk_id
* chunk_index
* chunk_text
* created_at
* doc_id
* file_name
* keywords (array<string>)
* section_id
* section_number

### 🔹 Purpose:

Stores section-based chunked text optimized for vector embedding generation.

This is the **main retrieval-ready dataset**.

---

# 🧠 6️⃣ Vector Storage (Chroma DB – File-Based)

You are using **Chroma persistent client**.

Two locations exist:

---

## 🔹 A. Primary Chroma Database

### Location:

```
/Volumes/workspace/legal_data/chroma_db/
```

Contains:

```
chroma.sqlite3
chroma.sqlite3-journal
```

These are:

* SQLite database files
* Used internally by Chroma
* Store collection metadata + embedding index

---

## 🔹 B. Secondary / Test Vector DB

### Location:

```
/Volumes/workspace/legal_data/vector_db/
```

Contains:

```
chroma.sqlite3
chroma.sqlite3-journal
```

Same purpose as above.

---

## 🔹 C. Test Knowledge Collection

### Location:

```
/Volumes/workspace/legal_data/vector_db_test/chroma_db_legal_knowledge_test/
```

Contains:

```
chroma.sqlite3
chroma.sqlite3-journal
```

This corresponds to:

```
collection = "legal_knowledge_test"
```

---

# 🧬 7️⃣ Embeddings Stored in Delta (Most Important)

This is your **main embedding storage layer**.

### Location:

```
/Volumes/workspace/legal_data/vector_db_test/legal_embeddings_delta/
```

### File Format:

Snappy compressed Parquet (Delta)

Example files:

```
part-00000-1443d7ba-37a1-4349-aea0-9bfe36c18bae.c000.snappy.parquet
part-00000-361f3162-3c50-4516-a9b9-a18b4c952042.c000.snappy.parquet
...
```

Each file ≈ 15.52 MB

### Purpose:

Stores:

* chunk_id
* metadata
* embedding vector (array<float>)
* associated chunk text

This is your **true vector data storage layer in Delta format**.

This is more scalable and production-ready compared to raw Chroma SQLite.

---

# 🗂 8️⃣ Complete Folder Structure Overview

```
/Volumes/workspace/legal_data/
│
├── bronze/
│   └── legal_documents/
│       └── Delta parquet files
│
├── silver/
│   └── legal_sections/
│       └── Delta parquet files
│
├── gold/
│   └── legal_chunks/
│       └── Delta parquet files
│
├── chroma_db/
│   └── chroma.sqlite3
│
├── vector_db/
│   └── chroma.sqlite3
│
├── vector_db_test/
│   ├── embedding_runtime_manifest.json
│   ├── chroma_db_legal_knowledge_test/
│   │   └── chroma.sqlite3
│   └── legal_embeddings_delta/
│       └── Delta parquet files (embeddings)
│
└── raw_documents/
```

---

# 🔄 9️⃣ Data Flow Summary

```
PDF Files
   ↓
Bronze (Raw Extracted Text)
   ↓
Silver (Section-Level Structured Data)
   ↓
Gold (Chunked Legal Text)
   ↓
Embeddings Generated
   ↓
Stored in:
   - Chroma SQLite DB
   - Delta table: legal_embeddings_delta
   ↓
Vector Retrieval for RAG
```

---

# 🏆 Final Important Clarification

Your **main vector storage for production should be:**

```
/Volumes/workspace/legal_data/vector_db_test/legal_embeddings_delta/
```

Because:

* It stores embeddings in scalable Delta format
* Not just SQLite metadata
* Better for Databricks Vector Search migration

The `chroma.sqlite3` files only store index metadata and small control data.

---

