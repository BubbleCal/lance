// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Run recall benchmarks for HNSW.
//!
//! run with `cargo run --release --example hnsw`

use arrow::array::AsArray;
use arrow::datatypes::Int32Type;
use arrow_array::types::Float32Type;
use clap::Parser;
use futures::TryStreamExt;
use lance::index::vector::VectorIndexParams;
use lance::Dataset;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::{DatasetIndexExt, IndexType};
use lance_linalg::distance::MetricType;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Dataset URI
    uri: String,

    /// Vector column name
    #[arg(short, long, value_name = "NAME", default_value = "vector")]
    column: Option<String>,

    #[arg(long, default_value = "100")]
    ef: usize,

    /// Max number of edges of each node.
    #[arg(long, default_value = "30")]
    max_edges: usize,

    #[arg(long, default_value = "7")]
    max_level: u16,

    #[arg(long, default_value = "1")]
    nprobe: usize,

    #[arg(short, default_value = "10")]
    k: usize,

    #[arg(long, default_value = "false")]
    create_index: bool,

    #[arg(long, default_value = "cosine")]
    metric_type: String,
}

#[cfg(test)]
fn ground_truth(mat: &MatrixView<Float32Type>, query: &[f32], k: usize) -> HashSet<u32> {
    let mut dists = vec![];
    for i in 0..mat.num_rows() {
        let dist = lance_linalg::distance::l2_distance(query, mat.row(i).unwrap());
        dists.push((dist, i as u32));
    }
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.truncate(k);
    dists.into_iter().map(|(_, i)| i).collect()
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let args = Args::parse();

    let mut dataset = Dataset::open(&args.uri)
        .await
        .expect("Failed to open dataset");
    println!("Dataset schema: {:#?}", dataset.schema());

    let column = args.column.as_deref().unwrap_or("vector");
    let metric_type = MetricType::try_from(args.metric_type.as_str()).unwrap();

    let mut ivf_params = IvfBuildParams::new(64);
    ivf_params.sample_rate = 20480;
    let hnsw_params = HnswBuildParams::default();
    let pq_params = SQBuildParams::default();
    let params =
        VectorIndexParams::with_ivf_hnsw_sq_params(metric_type, ivf_params, hnsw_params, pq_params);
    println!("{:?}", params);

    if args.create_index {
        let now = std::time::Instant::now();
        dataset
            .create_index(&[column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        println!("build={:.3}s", now.elapsed().as_secs_f32());
    }

    println!("Loaded {} records", dataset.count_rows(None).await.unwrap());

    let q = dataset
        .take(&[0], &dataset.schema().project(&[column]).unwrap())
        .await
        .unwrap()
        .column(0)
        .as_fixed_size_list()
        .values()
        .as_primitive::<Float32Type>()
        .clone();

    let columns: &[&str] = &[];
    let mut scan = dataset.scan();
    let plan = scan
        .project(columns)
        .unwrap()
        .with_row_id()
        .nearest(column, &q, args.k)
        .unwrap()
        .nprobs(args.nprobe)
        .filter("year >= 2000")
        .unwrap();
    println!("{:?}", plan.explain_plan(true).await.unwrap());

    let now = std::time::Instant::now();
    let results = plan
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    println!(
        "level={}, nprobe={}, k={}, search={:?}",
        args.max_level,
        args.nprobe,
        args.k,
        now.elapsed(),
    );
    // for r in results.iter() {
    //     let column = r
    //         .column_by_name("year")
    //         .unwrap()
    //         .as_primitive::<Int32Type>();
    //     for v in column.values() {
    //         assert!(*v >= 2000);
    //     }
    // }

    for concurrency in [1, 2, 4, 8, 12, 16, 24, 32] {
        // execute query with `concurrency` threads for 5 seconds and calculate qps and collect latencies
        let threads = (0..concurrency)
            .map(|_| {
                let column = column.to_owned();
                let q = q.clone();
                let dataset = dataset.clone();
                tokio::task::spawn(async move {
                    let mut scan = dataset.scan();
                    let plan = scan
                        .project(columns)
                        .unwrap()
                        .with_row_id()
                        .nearest(&column, &q, args.k)
                        .unwrap()
                        .nprobs(args.nprobe)
                        .filter("year >= 2000")
                        .unwrap();
                    let mut qps = 0;
                    let mut latencies = Vec::with_capacity(10000);
                    let now = std::time::Instant::now();
                    while now.elapsed().as_secs() < 5 {
                        let now = std::time::Instant::now();
                        plan.try_into_stream()
                            .await
                            .unwrap()
                            .try_collect::<Vec<_>>()
                            .await
                            .unwrap();
                        latencies.push(now.elapsed());
                        qps += 1;
                    }
                    (qps, latencies)
                })
            })
            .collect::<Vec<_>>();
        let mut qps = 0;
        let mut latencies = Vec::with_capacity(10000);
        for t in threads {
            let (q, l) = t.await.unwrap();
            qps += q;
            latencies.extend(l);
        }
        qps /= 5;
        latencies.sort();
        let p90 = latencies[latencies.len() * 90 / 100 - 1];
        let p95 = latencies[latencies.len() * 95 / 100 - 1];
        let p99 = latencies[latencies.len() * 99 / 100 - 1];
        let mean = latencies.iter().sum::<std::time::Duration>() / latencies.len() as u32;
        println!(
            "level={}, nprobe={}, k={}, concurrency={}, qps={}, mean={:?}, p90={:?}, p95={:?}, p99={:?}",
            args.max_level, args.nprobe, args.k, concurrency, qps, mean, p90, p95, p99,
        );
    }
}
