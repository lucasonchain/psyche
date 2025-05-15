## Psyche Data Provider – In-Depth Technical Report

_(revised with additional coverage and plain-English explanations)_

### Glossary (quick reference)
* **BatchId** – a closed integer interval identifying one or more **sequences** (contiguous blocks of tokens) to fetch.
* **Token** – an integer produced by a tokenizer; `TokenSize` is either `TwoBytes` (16-bit) or `FourBytes` (32-bit).
* **Sequence** – a fixed-length array of tokens plus one *next token* for language-model pre-training.
* **mmap** – *memory-mapping* (treating a file as if it were an in-memory byte array, no copy).
* **HTTP Range Request** – an HTTP feature that lets a client download only a slice of a file.
* **Shuffle::Seeded** – a deterministic pseudo-random shuffle powered by `ChaCha8Rng` (a fast cryptographically-strong random-number generator).
* **TCP** – Transmission Control Protocol (reliable socket connection between two machines).
* **Weight normalisation** – dividing weights by their sum so they add up to 1.

---

### 1. Repository Placement & Responsibility
The `shared/data-provider` crate is the **sole authority for training-data retrieval**.  Any component that needs training tokens (for example the `shared/client` training loop) depends only on two small *traits* – `TokenizedDataProvider` and `LengthKnownDataProvider`.  The actual storage back-end (local disk, HTTP, TCP, etc.) is hidden behind an **enum façade** called `DataProvider`.

This decouples the rest of the code-base from I/O details and makes it trivial to swap or compose new back-ends.

```
trainer / client ──▶  DataProvider   ──▶  (Local | HTTP | TCP | Weighted | Dummy)
```

---

### 2. Public Surface Area
1. **Traits (interface definitions)** – `src/traits.rs`
   * `TokenizedDataProvider` – exposes one async method `get_samples(&mut self, data_ids: BatchId)` which returns a `Vec<Vec<i32>>` (a batch of sequences, each sequence a list of token IDs).
   * `LengthKnownDataProvider` – reports how many sequences the provider can yield via `num_sequences()`.
2. **`DataProvider<T>` façade** – `src/data_provider.rs`
   * Variants: `Http`, `Server` (TCP client), `Dummy`, `WeightedHttp`.
   * Implements `TokenizedDataProvider` by simple `match` delegation.
3. **`lib.rs` re-exports** – Consumers can simply write `use psyche_data_provider::DataProvider;` and gain access to most items.

---

### 3. Concrete Providers (with internals)

#### 3.1 LocalDataProvider (`src/local.rs`)
* **Discovery** – scans a directory for extensions listed in `DATA_FILE_EXTENSIONS` (`.npy`, `.bin`, `.ds`).
* **Loading** – each file is `mmap`-ed (zero-copy) for fast random access.
* **Indexing** – builds a `Vec<SequencePointer>` (tuple of `file_index` + `byte_offset`).
* **Shuffle** – if the caller chose `Shuffle::Seeded(seed)` the index array is shuffled with `ChaCha8Rng(seed)` (deterministic RNG).
* **Retrieval path** – on `get_samples`: for every requested id it slices the mapped memory and converts bytes to `i32` tokens.

#### 3.2 HttpDataProvider (`src/http.rs`)
* **File catalogue** – `FileURLs` helper produces `(url, size)` pairs from:
  * a literal URL list,
  * a `{}` template with numeric substitution, or
  * a Google Cloud Storage (GCS) bucket listing via the *anonymous* GCS JSON API.
* **Pre-computation** – once file sizes are known the provider precomputes all possible **SequencePointer**s (same struct as local).
* **Fetching algorithm** – each call to `get_samples` issues one or more *HTTP Range Requests* per sequence; concurrency is achieved with `tokio::task::JoinHandle` and `join_all` (parallel futures collection).
* **Timeouts & retries** – `HTTP_REQUEST_TIMEOUT` (5 s) for each HTTP call; higher-level retries are handled by the `DataFetcher` in `shared/client`.

#### 3.3 Remote (TCP) Provider (`src/remote/*`)
* **Wire format** – messages `ClientToServerMessage` and `ServerToClientMessage` are `serde`-serialised (compact binary by default).
* **Security** – relies on `psyche_network::TcpClient`/`TcpServer` which implement *challenge-response* authentication using an `AuthenticatableIdentity` (public key + signature).
* **Coordinator integration** – the server owns a `Coordinator` (shared state of the federated run).  It only serves data if the requester is listed in the current round (`in_round` `HashSet`).
* **Observability** – stats are exposed in a Terminal UI (`DataServerTui`) showing samples served per client and overall progress.

#### 3.4 WeightedDataProvider (`src/weighted/mod.rs`)
* **Purpose** – merges several providers of the same concrete type into a single *virtual* dataset.
* **Weight sources** – user may supply explicit `(provider, weight)` pairs (**ExplicitlyWeighted**) or omit weights (**LengthWeighted**), in which case dataset lengths act as weights.
* **Algorithm** –
  1. For each provider compute target proportion `wᵢ`.
  2. Build primary vectors `dataset_index` and `dataset_sample_index` such that position *k* maps to one provider and its local sample.
  3. Use an **error-balancing sampling** loop (difference between *expected* and *actual* draw counts) to decide which provider fills the next slot.  This guarantees long-term proportions even when `n_samples` is not divisible by proportions.
  4. Optionally shuffle the vectors (again `ChaCha8Rng`).
* **Batch splitting optimisation** – When many contiguous samples originate from the same provider the code coalesces them into a single `get_samples` call, reducing round trips.

#### 3.5 DummyDataProvider (`src/dummy.rs`)
A no-I/O provider returning zeroed tokens – invaluable for unit tests, CI, and benchmarking the training loop without storage overhead.

---

### 4. Auxiliary Modules
* **Dataset (Parquet)** – `src/dataset.rs` loads HuggingFace-style Parquet datasets, detecting "train/test/validation" splits and optional *subset* folders.  Although not used by the tokenised providers, it is handy for analytics or re-tokenisation pipelines.
* **HF Hub Helpers** – `src/hub.rs` wraps the official `hf_hub` crate to:
  * asynchronously download model or dataset repos with extension filters, and
  * upload model files with commit metadata.
* **Examples CLI** – `examples/http.rs` and `examples/tcp.rs` demonstrate interactive use; the HTTP example is a mini *cURL-based* fetcher that can even decode samples with a provided tokenizer JSON.
* **Tests** – extensive property-based tests in `tests/weighted.rs` validate that the weighted sampler meets its statistical guarantees.

---

### 5. End-to-End Data Flow
1. **Coordinator** assigns batches → emits `data_assignments` map.
2. **DataFetcher** (`shared/client`) converts assignments into `BatchId`s.
3. **DataProvider** implementation retrieves bytes/tokens and returns `Vec<Vec<i32>>`.
4. **Batch** object travels to the GPU training loop.
5. Metrics (tokens/sec, samples served) are logged via `tracing`.

---

### 6. Strengths
* **Single abstraction layer** – greatly simplifies training code.
* **I/O efficiency** – memory-map or range-download only what is needed.
* **Reproducibility** – deterministic shuffling and weight normalisation.
* **Extensibility** – adding a new back-end is as simple as implementing the two traits.

### 7. Improvement Opportunities
1. **Typed error enums** (instead of generic `anyhow::Error`) for clearer caller handling.
2. **Request throttling** – a per-provider async semaphore would protect S3/GCS endpoints from excessive parallel requests.
3. **Streaming API** – returning an `async_stream::Stream<Item = Vec<i32>>` would allow back-pressure and lower peak RAM.
4. **Centralised metrics** – expose Prometheus-friendly counters for bytes read, latencies, and failures.
5. **Unified configuration format** – right now each provider has ad-hoc builders; a `psyche-data.yaml` could declaratively choose provider chains.

---

### 8. Quick-Start Cookbook
* **HTTP** –
  ```bash
  cargo run -p psyche_data_provider --example http -- \
    --sequence-length 2048 \
    --token-size 2 \
    --batch-ids 0,1,2 \
    template "https://my-bucket/{}.ds" --start 0 --end 100
  ```
* **Weighted HTTP** – provide a JSON config describing providers + weights, then:
  ```rust
  let provider = WeightedDataProvider::<HttpDataProvider>::from_config_url(
      "https://host/my_config.json", max_seq_len).await?;
  ```
* **TCP** – run `examples/tcp.rs` to spin up a local server and multiple clients.

---

### 9. Integrating a Custom REST API as a Data Provider
The crate is designed for easy extension. If your training data lives behind a bespoke REST (HTTP) service instead of static files, follow the steps below to plug it in.

1. **Design the REST contract**
   * **Endpoint shape** – The simplest pattern is `GET /sequences?start=<u64>&end=<u64>` that returns a contiguous range of token IDs.
   * **Binary vs JSON** – For maximum throughput return a binary blob (little-endian `u16`/`u32`).  If JSON is unavoidable, be aware of the extra CPU / bandwidth.
   * **Authentication** – Re-use existing bearer-token headers or mTLS—anything supported by `reqwest`.

2. **Create a thin wrapper struct**
   ```rust
   use anyhow::Result;
   use psyche_core::{BatchId, TokenSize};
   use psyche_data_provider::{LengthKnownDataProvider, TokenizedDataProvider};

   pub struct MyApiDataProvider {
       client: reqwest::Client,
       base_url: String,
       seq_len: u32,
       token_size: TokenSize,
       total_sequences: usize, // cache size if the API exposes it
   }
   ```

3. **Implement `LengthKnownDataProvider`**
   ```rust
   impl LengthKnownDataProvider for MyApiDataProvider {
       fn num_sequences(&self) -> usize {
           self.total_sequences
       }
   }
   ```

4. **Implement `TokenizedDataProvider`**
   ```rust
   #[async_trait::async_trait]
   impl TokenizedDataProvider for MyApiDataProvider {
       async fn get_samples(&mut self, data_ids: BatchId) -> Result<Vec<Vec<i32>>> {
           // Construct request
           let url = format!(
               "{}/sequences?start={}&end={}",
               self.base_url,
               data_ids.start(),
               data_ids.end()
           );
           // Fetch bytes
           let bytes = self.client.get(url).send().await?.bytes().await?;
           // Convert to Vec<i32>
           let tokens: Vec<i32> = bytes
               .chunks(self.token_size.into())
               .map(|chunk| match self.token_size {
                   TokenSize::TwoBytes => u16::from_le_bytes(chunk.try_into().unwrap()) as i32,
                   TokenSize::FourBytes => u32::from_le_bytes(chunk.try_into().unwrap()) as i32,
               })
               .collect();
           Ok(vec![tokens])
       }
   }
   ```

5. **(Optional) Expose through the façade**
   * Add a new variant to `enum DataProvider`:
     ```rust
     Api(MyApiDataProvider),
     ```
   * Update the `match` block in its `TokenizedDataProvider` impl:
     ```rust
     DataProvider::Api(p) => p.get_samples(data_ids).await,
     ```
   * Re-export the new type in `lib.rs`.

6. **Use it in `shared/client`**
   ```rust
   let api_provider = MyApiDataProvider {
       client: reqwest::Client::new(),
       base_url: "https://data.mycorp.com".into(),
       seq_len: 2048,
       token_size: TokenSize::TwoBytes,
       total_sequences: /* fetch from /meta or similar */ 1_000_000,
   };
   let mut data_provider = DataProvider::Api(api_provider);
   ```

7. **Performance tips**
   * **HTTP keep-alive** – `reqwest::Client` does this by default; avoid constructing a new client per call.
   * **Compression** – If network-bound, enable `br` (Brotli) or `gzip` and decompress in memory.
   * **Batch ranges** – Wider ranges amortise latency but increase memory; tune to match GPU batch size.

Following these steps lets you consume any internal data service while keeping the broader *Psyche* ecosystem unchanged.

---

### 10. Final Verdict
The *Psyche* Data Provider subsystem is a **robust, highly modular, and production-ready** piece of infrastructure that abstracts away the messiness of heterogeneous storage layers.  With a few targeted improvements in ergonomics and observability it can scale to very large, multi-tenant training scenarios.


# data-provider

there's a bunch of functionality here, but the http stuff is what you probably wanna try out.

## http data provider fetch example

### Usage

#### working example

First, an example:
`cargo run --example http -- --file-size 40000004052 --batch-ids 103 --token-size 4 --tokenizer tests/resources/llama3_tokenizer.json urls https://storage.googleapis.com/nous-pretraining-public-us/fineweb-1pct-tokenized-llama3/000_fineweb.ds`

This will fetch some fineweb data & output it using the llama3 tokenizer!

#### Basic Command Structure

```bash
cargo run --example http --file-size <SIZE> [--sequence-length <LENGTH>] [--token-size <SIZE>] --batch-ids <IDS> [--tokenizer <PATH>] <SUBCOMMAND>
```

The tool supports two main modes of operation: template-based URLs and explicit URL lists.

#### Required

- `--batch-ids`: Comma-separated list of batch IDs to retrieve

#### Optional

- `--sequence-length`: Length of each sequence (default: 2048)
- `--token-size`: Size of each token in bytes (default: 2)
- `--tokenizer`: Path to tokenizer file for decoding output

#### Subcommands

##### Template Mode

```bash
template <TEMPLATE> --start <START> --end <END> [--left-pad-zeros <N> (default 0)]
```

Example:

```bash
cargo run --example http --batch-ids 1,2,3 template "http://example.com/{}.ds" --start 0 --end 10
```

this will fetch urls http://example.com/0.ds thru http://example.com/10.ds

###### left pad zeros

`--left-pad-zeros 3` will transform fetch URLs http://example.com/000.ds thru http://example.com/010.ds

##### URL List Mode

```bash
urls <URL1> <URL2> ...
```

Example:

```bash
cargo run --example http --batch-ids 1,2,3 urls "http://example.com/1.ds" "http://example.com/2.ds"
```

### Examples

1. Fetch data using a template with tokenizer:

```bash
cargo run --example http --batch-ids 1,2,3 --tokenizer ./tokenizer.json template "http://example.com/{}.ds" --start 0 --end 10
```

2. Fetch data using explicit URLs:

```bash
cargo run --example http --sequence-length 1024 --batch-ids 1,2,3 urls "http://example.com/data1.ds" "http://example.com/data2.ds"
```

### Output

The tool will output the retrieved samples for each batch ID. If a tokenizer is specified, the output will be decoded using the tokenizer. Otherwise, the raw sample data will be displayed.
