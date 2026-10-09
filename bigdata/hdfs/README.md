# Hadoop Distributed File System (HDFS) — Conceptual Architecture

## Academic Architecture Blueprint

This document details the conceptual mapping of **MahaTraffic AI** datasets onto an enterprise **Hadoop Distributed File System (HDFS)** cluster.

> ⚠️ **Implementation Notice:**  
> In accordance with project requirements prioritizing reproducibility and demonstrable execution in local VS Code environments on Windows, **HDFS is NOT a required runtime dependency**.  
> The working implementation utilizes local filesystem persistence with **columnar Apache Parquet (PyArrow)** storage.

---

## 1. What is HDFS?
HDFS is a distributed, scalable file system designed to run on commodity hardware. It provides high-throughput access to application data, fault tolerance through block replication (default 3x), and scales across petabyte data lakes.

---

## 2. Conceptual Namespace Mapping in HDFS
In a full-scale production deployment, the data hierarchy would map to the HDFS namespace as follows:

```text
hdfs://namenode:9000/mahatraffic/
├── raw/
│   ├── accidents/
│   │   └── maharashtra_district_accidents_2019_2023.csv
│   └── social_media/
│       └── maharashtra_road_safety_public_posts.csv
├── processed/
│   ├── parquet/
│   │   ├── accidents/
│   │   │   ├── year=2019/
│   │   │   ├── year=2020/
│   │   │   ├── year=2021/
│   │   │   ├── year=2022/
│   │   │   └── year=2023/
│   │   └── social_media/
│   └── features/
│       └── risk_features.parquet
```

---

## 3. Why Local Filesystem + Parquet is Used
1. **Low Overhead & Zero Daemon Latency:** Eliminates the need for NameNode/DataNode Java services, ZooKeeper, and YARN resource managers on a developer workstation.
2. **Identical Columnar Benefits:** Local `.parquet` files implement the identical Google Dremel record shredding format utilized by HDFS-backed Spark and Hive.
3. **Seamless Transition:** PySpark code (`spark.read.parquet("hdfs://...")` vs `spark.read.parquet("data/...")`) requires changing only the URI scheme.
