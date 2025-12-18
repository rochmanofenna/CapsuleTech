use rand::Rng;
use std::time::Instant;

use ark_bls12_381::Bls12_381;
use ark_poly::univariate::DensePolynomial;
use ark_poly::DenseUVPolynomial;
use ark_poly_commit::kzg10::{Powers, VerifierKey, KZG10};
use ark_ff::UniformRand;
use ark_std::test_rng;

const NS: [usize; 3] = [1 << 20, 1 << 24, 1 << 26];
const REPEATS: usize = 3;

fn setup(degree: usize) -> (Powers<Bls12_381>, VerifierKey<Bls12_381>) {
    let mut rng = test_rng();
    KZG10::<Bls12_381, DensePolynomial<_>>::setup(degree, false, &mut rng).expect("setup")
}

fn bench_kzg_commit(n: usize, powers: &Powers<Bls12_381>) -> f64 {
    let mut rng = test_rng();
    let coeffs: Vec<_> = (0..n).map(|_| ark_bls12_381::Fr::rand(&mut rng)).collect();
    let poly = DensePolynomial::from_coefficients_vec(coeffs);
    let start = Instant::now();
    let l0 = KZG10::commit(powers, &poly, None, None).expect("commit");
    let _commitment = l0.0;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    elapsed_ms
}

fn main() {
    println!("N,commit_ms,throughput_mb_s");
    for &n in &NS {
        let degree = n.next_power_of_two();
        let (powers, _) = setup(degree);
        let mut best_ms = f64::INFINITY;
        for _ in 0..REPEATS {
            let ms = bench_kzg_commit(n, &powers);
            if ms < best_ms {
                best_ms = ms;
            }
        }
        let bytes = (n as f64) * 32.0;
        let throughput_mb_s = bytes / (best_ms / 1_000.0) / (1024.0 * 1024.0);
        println!("{}, {:.3}, {:.3}", n, best_ms, throughput_mb_s);
    }
}
