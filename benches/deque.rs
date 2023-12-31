#![allow(unused)]

use criterion::{criterion_group, criterion_main, Criterion};
use crossbeam::deque::{Steal, Worker as CrossbeamWorker};
use viscose::deque::Worker as ViscoseWorker;

fn criterion_benchmark(c: &mut Criterion) {
    #[cfg(feature = "bench-deque")]
    {
        /// Element has same size/layout as a Task
        type Element = (usize, usize, u8);

        /// Queue fits in the L2 cache
        const VISCOSE_CAPACITY: usize = 128_000 / std::mem::size_of::<Element>();

        // Set up the basic benchmark
        let mut crossbeam_local = CrossbeamWorker::new_lifo();
        let crossbeam_remote = crossbeam_local.stealer();
        let mut viscose_local = ViscoseWorker::new(VISCOSE_CAPACITY);
        let viscose_remote = viscose_local.remote();
        let element = Element::default();

        // Benchmark basic operations in a single uncontended thread
        {
            let mut group = c.benchmark_group("uncontended");
            group.bench_function("push+pop/crossbeam", |b| {
                b.iter(|| {
                    crossbeam_local.push(pessimize::hide(element));
                    pessimize::consume(&crossbeam_local.pop());
                })
            });
            group.bench_function("push+pop/viscose", |b| {
                b.iter(|| {
                    let _ = viscose_local.push(pessimize::hide(element));
                    pessimize::consume(&viscose_local.pop());
                })
            });
            group.bench_function("push+steal/crossbeam", |b| {
                b.iter(|| {
                    crossbeam_local.push(pessimize::hide(element));
                    pessimize::consume(&crossbeam_remote.steal());
                })
            });
            group.bench_function("push+steal/viscose", |b| {
                b.iter(|| {
                    let _ = viscose_local.push(pessimize::hide(element));
                    pessimize::consume(&viscose_remote.steal());
                })
            });
            group.bench_function("give+pop/viscose", |b| {
                b.iter(|| {
                    let _ = viscose_remote.give(pessimize::hide(element));
                    pessimize::consume(&viscose_local.pop());
                })
            });
            group.bench_function("give+steal/viscose", |b| {
                b.iter(|| {
                    let _ = viscose_remote.give(pessimize::hide(element));
                    pessimize::consume(&viscose_remote.steal());
                })
            });
        }

        // Benchmark push operations under contention
        {
            let mut group = c.benchmark_group("contended/push/stolen");
            {
                let group = &mut group;
                let viscose_local = &mut viscose_local;
                testbench::run_under_contention(
                    || pessimize::consume(&viscose_remote.steal()),
                    move || {
                        group.bench_function("try/viscose", |b| {
                            b.iter(|| {
                                viscose_local.push(pessimize::hide(element));
                            })
                        });
                    },
                );
            }
            {
                let group = &mut group;
                let crossbeam_local = &mut crossbeam_local;
                testbench::run_under_contention(
                    || pessimize::consume(&crossbeam_remote.steal()),
                    move || {
                        group.bench_function("force/crossbeam", |b| {
                            b.iter(|| {
                                crossbeam_local.push(pessimize::hide(element));
                            })
                        });
                    },
                );
            }
            {
                let group = &mut group;
                let viscose_local = &mut viscose_local;
                testbench::run_under_contention(
                    || pessimize::consume(&viscose_remote.steal()),
                    move || {
                        group.bench_function("force/viscose", |b| {
                            b.iter(
                                || {
                                    while viscose_local.push(pessimize::hide(element)) != Ok(()) {}
                                },
                            )
                        });
                    },
                );
            }
        }

        // Benchmark pop operations under contention
        {
            let mut group = c.benchmark_group("contended/pop/given");
            {
                let group = &mut group;
                let viscose_local = &mut viscose_local;
                testbench::run_under_contention(
                    || viscose_remote.give(pessimize::hide(element)),
                    move || {
                        group.bench_function("try/viscose", |b| {
                            b.iter(|| {
                                pessimize::consume(&viscose_local.pop());
                            })
                        });
                    },
                );
            }
            {
                let group = &mut group;
                let viscose_local = &mut viscose_local;
                testbench::run_under_contention(
                    || viscose_remote.give(pessimize::hide(element)),
                    move || {
                        group.bench_function("force/viscose", |b| {
                            b.iter(|| loop {
                                match viscose_local.pop() {
                                    Some(elem) => {
                                        pessimize::consume(elem);
                                        break;
                                    }
                                    None => continue,
                                }
                            })
                        });
                    },
                );
            }
        }

        // Benchmark steal operations under contention
        {
            let mut group = c.benchmark_group("contended/steal");
            {
                let group = &mut group;
                let crossbeam_local = &mut crossbeam_local;
                testbench::run_under_contention(
                    move || crossbeam_local.push(pessimize::hide(element)),
                    || {
                        group.bench_function("pushed/try/crossbeam", |b| {
                            b.iter(|| {
                                pessimize::consume(&crossbeam_remote.steal());
                            })
                        });
                    },
                );
            }
            {
                let group = &mut group;
                let viscose_local = &mut viscose_local;
                testbench::run_under_contention(
                    move || viscose_local.push(pessimize::hide(element)),
                    || {
                        group.bench_function("pushed/try/viscose", |b| {
                            b.iter(|| {
                                pessimize::consume(&viscose_remote.steal());
                            })
                        });
                    },
                );
            }
            {
                let group = &mut group;
                let crossbeam_local = &mut crossbeam_local;
                testbench::run_under_contention(
                    move || crossbeam_local.push(pessimize::hide(element)),
                    || {
                        group.bench_function("pushed/force/crossbeam", |b| {
                            b.iter(|| loop {
                                match crossbeam_remote.steal() {
                                    Steal::Success(elem) => {
                                        pessimize::consume(elem);
                                        break;
                                    }
                                    _ => continue,
                                }
                            })
                        });
                    },
                );
            }
            {
                let group = &mut group;
                let viscose_local = &mut viscose_local;
                testbench::run_under_contention(
                    move || viscose_local.push(pessimize::hide(element)),
                    || {
                        group.bench_function("pushed/force/viscose", |b| {
                            b.iter(|| loop {
                                match viscose_remote.steal() {
                                    Ok(elem) => {
                                        pessimize::consume(elem);
                                        break;
                                    }
                                    _ => continue,
                                }
                            })
                        });
                    },
                );
            }
            {
                let group = &mut group;
                testbench::run_under_contention(
                    || viscose_remote.give(pessimize::hide(element)),
                    || {
                        group.bench_function("given/try/viscose", |b| {
                            b.iter(|| {
                                pessimize::consume(&viscose_remote.steal());
                            })
                        });
                    },
                );
            }
            {
                let group = &mut group;
                testbench::run_under_contention(
                    || viscose_remote.give(pessimize::hide(element)),
                    || {
                        group.bench_function("given/force/viscose", |b| {
                            b.iter(|| loop {
                                match viscose_remote.steal() {
                                    Ok(elem) => {
                                        pessimize::consume(elem);
                                        break;
                                    }
                                    _ => continue,
                                }
                            })
                        });
                    },
                );
            }
        }

        // Benchmark give operations under contention
        {
            let mut group = c.benchmark_group("contended/give");
            {
                let group = &mut group;
                let viscose_local = &mut viscose_local;
                testbench::run_under_contention(
                    move || pessimize::consume(&viscose_local.pop()),
                    || {
                        group.bench_function("popped/try/viscose", |b| {
                            b.iter(|| {
                                viscose_remote.give(pessimize::hide(element));
                            })
                        });
                    },
                );
            }
            {
                let group = &mut group;
                let viscose_local = &mut viscose_local;
                testbench::run_under_contention(
                    move || pessimize::consume(&viscose_local.pop()),
                    || {
                        group.bench_function("popped/force/viscose", |b| {
                            b.iter(
                                || {
                                    while viscose_remote.give(pessimize::hide(element)) != Ok(()) {}
                                },
                            )
                        });
                    },
                );
            }
            {
                let group = &mut group;
                testbench::run_under_contention(
                    || pessimize::consume(&viscose_remote.steal()),
                    || {
                        group.bench_function("stolen/try/viscose", |b| {
                            b.iter(|| {
                                viscose_remote.give(pessimize::hide(element));
                            })
                        });
                    },
                );
            }
            {
                let group = &mut group;
                testbench::run_under_contention(
                    || pessimize::consume(&viscose_remote.steal()),
                    || {
                        group.bench_function("stolen/force/viscose", |b| {
                            b.iter(
                                || {
                                    while viscose_remote.give(pessimize::hide(element)) != Ok(()) {}
                                },
                            )
                        });
                    },
                );
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
