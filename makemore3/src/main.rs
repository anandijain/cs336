use std::{
    collections::{BTreeSet, HashMap},
    fs, iter,
};

use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{
    Distribution,
    weighted::{Weight, WeightedIndex},
};

fn main() {
    let s = fs::read_to_string("names.txt").unwrap();
    let ws: Vec<_> = s.lines().collect();
    let mut alpha: Vec<_> = ('a'..='z').collect();
    alpha.push('.');
    let itos: HashMap<usize, char> = alpha.iter().copied().enumerate().collect();
    let stoi: HashMap<char, usize> = alpha
        .iter()
        .copied()
        .enumerate()
        .map(|(i, x)| (x, i))
        .collect();
    let mut counts = Array2::<u32>::zeros((alpha.len(), alpha.len()));
    for w in &ws {
        let mut chars = iter::once('.').chain(w.chars()).chain(iter::once('.'));
        let mut prev = chars.next().unwrap();

        for ch in chars {
            counts[[stoi[&prev], stoi[&ch]]] += 1;
            prev = ch
        }
    }
    let mut probs = counts.mapv(|x| x as f64);
    probs.rows_mut().into_iter().for_each(|mut row| {
        let row_sum = row.sum();
        row.mapv_inplace(|x| x / row_sum)
    });
    println!("{counts:?}");
    let mut rng = StdRng::seed_from_u64(42);
    let dists: Vec<_> = counts
        .rows()
        .into_iter()
        .map(|r| WeightedIndex::new(r).unwrap())
        .collect();
    for i in 0..10 {
        let mut cur = '.';
        loop {
            cur = itos[&dists[stoi[&cur]].sample(&mut rng)];
            print!("{cur}");
            if cur == '.' {
                println!();
                break;
            }
        }
    }
    let zero_bigrams: Vec<_> = counts.indexed_iter().filter_map(|((i, j), &count)| {
        if count == 0 {
            Some((alpha[i], alpha[j]))
        } else {
            None
        }
    }).collect();
    println!("zero_bigrams: {zero_bigrams:?}");
    let p = word_probability(&probs, &stoi, "bf");
    println!("prob of anand: {p}");
}

fn wrap_dots<'a>(s: &'a str) -> impl Iterator<Item = char> + 'a {
    iter::once('.').chain(s.chars()).chain(iter::once('.'))
}

fn word_probability(probs: &Array2<f64>, stoi: &HashMap<char, usize>, s: &str) -> f64 {
    let mut p = 1.0;
    let str = s.to_string();
    let mut chars = wrap_dots(&s);
    let mut prev = chars.next().unwrap();
    for ch in chars {
        p *= probs[[stoi[&prev], stoi[&ch]]];
        prev = ch;
    }
    p
}
