// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of HNSW graph.
//!
//!

use std::{sync::Arc, time::Duration};

use arrow_array::{RecordBatch, StringArray, UInt64Array};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use itertools::Itertools;
use lance_core::ROW_ID;
use lance_index::scalar::inverted::InvertedIndex;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{ScalarIndex, ScalarQuery};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;

#[tokio::main]
async fn main() {
    const TOTAL: usize = 30_000_000;

    let tempdir = tempfile::tempdir().unwrap();
    let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
    let store = Arc::new(LanceIndexStore::new(ObjectStore::local(), index_dir, None));

    let invert_index = InvertedIndex::default();
    let all_genres = [
        "Action",
        "Adventure",
        "Comedy",
        "Drama",
        "Fantasy",
        "Horror",
        "Mystery",
        "Romance",
        "Fiction",
        "Thriller",
        "Western",
        "Documentary",
        "Animation",
    ];
    let row_id_col = Arc::new(UInt64Array::from(
        (0..TOTAL).map(|i| i as u64).collect_vec(),
    ));
    let docs = (0..TOTAL)
        .map(|_| {
            let num_genres = rand::random::<usize>() % 3 + 1;
            let genres = (0..num_genres)
                .map(|_| all_genres[rand::random::<usize>() % all_genres.len()])
                .collect::<Vec<_>>();
            genres.join(" ")
        })
        .collect_vec();
    let doc_col = Arc::new(StringArray::from(docs));
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("doc", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
        ])
        .into(),
        vec![doc_col.clone(), row_id_col.clone()],
    )
    .unwrap();
    let stream = RecordBatchStreamAdapter::new(batch.schema(), stream::iter(vec![Ok(batch)]));
    let stream = Box::pin(stream);

    invert_index.update(stream, store.as_ref()).await.unwrap();
    let invert_index = InvertedIndex::load(store).await.unwrap();

    for concurrency in [1, 2, 4, 8, 16] {
        let threads = (0..concurrency)
            .map(|_| {
                let invert_index = invert_index.clone();
                let mut qps = 0;
                let mut latencies = Vec::with_capacity(50000);
                let query = ScalarQuery::FullTextSearch(vec!["Action".to_string()]);
                tokio::spawn(async move {
                    let start = std::time::Instant::now();
                    while start.elapsed() < Duration::from_secs(5) {
                        let start = std::time::Instant::now();
                        let res = invert_index.search(&query).await.unwrap();
                        let elapsed = start.elapsed();
                        latencies.push(elapsed);
                        qps += 1;
                    }
                    (qps, latencies)
                })
            })
            .collect::<Vec<_>>();

        let mut qps = 0;
        let mut latencies = Vec::new();
        for thread in threads {
            let (q, l) = thread.await.unwrap();
            qps += q;
            latencies.extend(l);
        }
        qps /= 5;
        latencies.sort();
        let p50 = latencies[latencies.len() * 50 / 100 - 1];
        let p90 = latencies[latencies.len() * 90 / 100 - 1];
        let p99 = latencies[latencies.len() * 99 / 100 - 1];
        println!(
            "concurrency: {}, qps: {}, p50: {:?}, p90: {:?}, p99: {:?}",
            concurrency, qps, p50, p90, p99
        );
    }
}
