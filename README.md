**This repo is abandoned, but I'll wait for about a year (til August 2025) before
archiving it so that any interested rayon contributor can ask questions via issues**

# Viscose: Prototyping locality-aware fork-join task scheduling for rayon

This project was created to answer [one
question](https://github.com/rayon-rs/rayon/issues/319#issuecomment-1783731222):
what would be the costs and benefits of making the task scheduling of
[rayon](https://docs.rs/rayon/latest/rayon/) NUMA-aware, and how would the
associated code look like in practice?

Other questions that come up along the way (most recently "What is the very 
costly `mfence` x86 instruction doing in the the generated assembly for
[`crossbeam::deque`](https://docs.rs/crossbeam/latest/crossbeam/deque/index.html)'s
hot code path?") will hopefully also be answered. But in my ideal world, this
repo will never become a dedicated library on crates.io, rather its ideas will
eventually be integrated into other relevant libraries in the Rust ecosystem.
