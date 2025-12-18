use bicep_core::{State, Calc, SdeIntegrator, Drift, Diffusion};
use bicep_core::integrators::EulerMaruyama;
use bicep_core::noise::NoiseGenerator;
use nalgebra::DMatrix;
use clap::Parser;
use anyhow::Result;
use polars::prelude::*;
use std::path::PathBuf;
use rand::{Rng, SeedableRng};

#[derive(Parser, Debug)]
#[command(author, version, about = "Generate parity task trajectories for ENN training")]
struct Args {
    #[arg(long, default_value_t = 1000)]
    sequences: usize,
    
    #[arg(long, default_value_t = 15)]
    seq_len: usize,
    
    #[arg(long, default_value_t = 1e-2)]
    dt: f64,
    
    #[arg(long, default_value = "runs/parity_trajectories.parquet")]
    out: PathBuf,
    
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

// Parity task: Generate binary sequences with XOR parity target
struct ParityTask {
    noise_level: f64,
}

impl Drift for ParityTask {
    fn mu(&self, _t: f64, x: &State) -> State {
        // Light drift toward binary values {-1, 1}
        let drift = x.0[0].tanh() - x.0[0];
        State::new(vec![drift * 0.1])
    }
}

impl Diffusion for ParityTask {
    fn sigma(&self, _t: f64, _x: &State) -> DMatrix<f64> {
        DMatrix::from_element(1, 1, self.noise_level)
    }
    
    fn sigma_jacobian(&self, _t: f64, _x: &State) -> Option<Vec<DMatrix<f64>>> {
        Some(vec![DMatrix::from_element(1, 1, 0.0)])
    }
    
    fn noise_dim(&self, _t: f64, _x: &State) -> usize {
        1
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    let model = ParityTask { noise_level: 0.5 };
    let integrator = EulerMaruyama;
    
    println!("Generating {} parity task sequences", args.sequences);
    println!("Sequence length: {}, dt: {}", args.seq_len, args.dt);
    
    // Data collectors
    let mut run_ids = Vec::new();
    let mut seeds = Vec::new();
    let mut models = Vec::new();
    let mut calcs = Vec::new();
    let mut dts = Vec::new();
    let mut seq_lens = Vec::new();
    let mut sequence_ids = Vec::new();
    let mut step_nums = Vec::new();
    let mut times = Vec::new();
    let mut states = Vec::new();
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    
    // Generate parity sequences
    for seq_id in 0..args.sequences {
        if seq_id % 100 == 0 {
            println!("Progress: {}/{} sequences", seq_id, args.sequences);
        }
        
        // Generate random binary sequence
        let binary_seq: Vec<i32> = (0..args.seq_len)
            .map(|_| if rng.gen::<bool>() { 1 } else { 0 })
            .collect();
        
        // Compute XOR parity (1 if odd number of 1s, 0 if even)
        let parity = binary_seq.iter().sum::<i32>() % 2;
        
        // Convert to floating point inputs {-1, 1} for SDE simulation
        let float_seq: Vec<f64> = binary_seq.iter()
            .map(|&x| if x == 1 { 1.0 } else { -1.0 })
            .collect();
        
        // Simulate each input as a short trajectory to add temporal structure
        let mut noise_gen = NoiseGenerator::from_path_id(args.seed, seq_id as u64);
        
        for (step, &input_val) in float_seq.iter().enumerate() {
            let mut state = State::new(vec![input_val + rng.gen_range(-0.1..0.1)]); // Add small noise
            let mut t = step as f64 * args.dt;
            
            // Short SDE evolution to add trajectory structure
            for _ in 0..5 {
                let dw = noise_gen.generate_dw(1, (args.dt * 0.1).sqrt());
                state = integrator.step(Calc::Ito, t, &state, args.dt * 0.1, &dw, &model, &model);
                t += args.dt * 0.1;
            }
            
            // Store the data
            run_ids.push("parity_task".to_string());
            seeds.push(args.seed);
            models.push("ParityTask".to_string());
            calcs.push("Ito".to_string());
            dts.push(args.dt);
            seq_lens.push(args.seq_len as u32);
            sequence_ids.push(seq_id as u64);
            step_nums.push(step as u32);
            times.push(step as f64 * args.dt);
            states.push(vec![state.0[0]]);
            inputs.push(input_val);
            
            // Target is only revealed at the end
            if step == args.seq_len - 1 {
                targets.push(parity as f64);
            } else {
                targets.push(0.0); // No target during sequence
            }
        }
    }
    
    // Create DataFrame
    let state_series = Series::new("state", 
        states.into_iter()
            .map(|v| Series::new("", v))
            .collect::<Vec<_>>()
    );
    
    let df = DataFrame::new(vec![
        Series::new("run_id", run_ids),
        Series::new("seed", seeds),
        Series::new("model", models),
        Series::new("calc", calcs),
        Series::new("dt", dts),
        Series::new("seq_len", seq_lens),
        Series::new("sequence_id", sequence_ids),
        Series::new("step", step_nums),
        Series::new("t", times),
        state_series,
        Series::new("input", inputs),
        Series::new("target", targets),
    ])?;
    
    // Save to Parquet
    std::fs::create_dir_all(args.out.parent().unwrap())?;
    let mut file = std::fs::File::create(&args.out)?;
    ParquetWriter::new(&mut file).finish(&mut df.clone())?;
    
    println!("Saved {} rows to {}", df.height(), args.out.display());
    
    // Print statistics
    let final_df = df.lazy()
        .group_by([col("sequence_id")])
        .agg([
            col("target").last().alias("final_target"),
            col("input").sum().alias("input_sum"),
        ])
        .collect()?;
    
    let n_positive = final_df.column("final_target")?.f64()?.sum().unwrap_or(0.0) as usize;
    let n_negative = args.sequences - n_positive;
    
    println!("\nParity task statistics:");
    println!("Sequences with parity 1: {} ({:.1}%)", n_positive, 100.0 * n_positive as f64 / args.sequences as f64);
    println!("Sequences with parity 0: {} ({:.1}%)", n_negative, 100.0 * n_negative as f64 / args.sequences as f64);
    
    Ok(())
}