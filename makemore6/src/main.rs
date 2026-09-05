use std::{collections::HashMap, fs, iter::once};

use ndarray::prelude::*;
use ndarray::{Array2, Axis};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, weighted::WeightedIndex};

fn wrap_dots(s: &str) -> impl Iterator<Item = char> {
    once('.').chain(s.chars()).chain(once('.'))
}

fn generate_name(
    dists: &Vec<WeightedIndex<usize>>,
    itos: &HashMap<usize, char>,
    rng: &mut StdRng,
) -> String {
    let mut cur = 0;
    let mut name_chars = vec![];
    loop {
        cur = dists[cur].sample(rng);
        if cur == 0 {
            break;
        }
        name_chars.push(cur);
    }
    name_chars.iter().map(|c| itos[c]).collect()
}

fn word_probablity(probs: &Array2<f64>, stoi: &HashMap<char, usize>, s: &str) -> f64 {
    // let mut chars = wrap_dots(s);
    // let mut prev = chars.next().unwrap();
    // let mut p = 1.0;
    // for ch in chars {
    //     let idx = (stoi[&prev], stoi[&ch]);
    //     p *= probs[idx];
    // }
    // p

    wrap_dots(s)
        .zip(wrap_dots(s).skip(1))
        .map(|(prev, cur)| probs[[stoi[&prev], stoi[&cur]]])
        .product()
}

fn main() {
    let s = fs::read_to_string("names.txt").unwrap();
    let alpha: Vec<_> = once('.').chain('a'..='z').collect();

    let itos: HashMap<usize, char> = alpha.iter().copied().enumerate().collect();
    let stoi: HashMap<char, usize> = alpha.iter().enumerate().map(|(i, &x)| (x, i)).collect();

    let n_alpha = alpha.len();
    // println!("{alpha:?}");
    let mut counts = Array2::<usize>::zeros((n_alpha, n_alpha));
    for n in s.lines() {
        let mut chars = once('.').chain(n.chars()).chain(once('.'));
        let mut prev = chars.next().unwrap();
        for ch in chars {
            let idx = [prev, ch].map(|c| stoi[&c]);
            counts[idx] += 1;
            prev = ch;
        }
    }
    // counbts/WeightedIndex::new()
    let dists: Vec<_> = counts
        .rows()
        .into_iter()
        .map(|row| WeightedIndex::new(row).unwrap())
        .collect();
    let mut rng = StdRng::seed_from_u64(42);

    // let probs: Array2<f64> = counts
    //     .rows()
    //     .into_iter()
    //     .map(|row| {
    //         let sum = row.sum() as f64;
    //         row.mapv(|x| (x as f64) / sum)
    //     })
    //     .collect();

    // let mut probs = counts.mapv(|x| x as f64);
    // probs.rows_mut().into_iter().for_each(|mut row| {
    //     let sum = row.sum();
    //     row.mapv_inplace(|x| x / sum);
    // });

    let mut probs = counts.mapv(|x| x as f64);
    for mut row in probs.rows_mut() {
        let sum = row.sum();
        row.mapv_inplace(|x| x / sum);
    }

    let mut p2 = counts.mapv(|x| x as f64);
    let row_sums = p2.sum_axis(Axis(1)).insert_axis(Axis(1)); // (alpha, 1)
    p2 /= &row_sums;

    let name = generate_name(&dists, &itos, &mut rng);
    println!("{:?}", word_probablity(&probs, &stoi, "anand"));
}
