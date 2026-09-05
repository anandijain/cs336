use std::{collections::HashMap, fs, iter::once};

use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, weighted::WeightedIndex};

fn wrap_dots<'a>(s: &'a str) -> impl Iterator<Item = char> {
    once('.').chain(s.chars()).chain(once('.'))
}

fn generate_name(
    mut rng: &mut StdRng,
    dists: &Vec<WeightedIndex<usize>>,
    itos: &HashMap<usize, char>,
) -> String {
    let mut cur = 0;
    let mut word = vec![];
    loop {
        cur = dists[cur].sample(&mut rng);
        if cur == 0 {
            break;
        }
        word.push(cur);
    }
    word.iter().map(|i| itos[i]).collect()
}

fn word_probability(s: &str, probs: &Array2<f64>, stoi: &HashMap<char, usize>) -> f64 {
    // let mut p = 1.0;
    // let mut chars = wrap_dots(s);
    // let mut prev = chars.next().unwrap();
    // for ch in chars {
    //     p *= probs[[stoi[&prev], stoi[&ch]]];
    //     prev = ch;
    // }
    // chars.collect()
    // p
    wrap_dots(s)
        .zip(wrap_dots(s).skip(1))
        .map(|(prev, cur)| probs[[stoi[&prev], stoi[&cur]]])
        .product()
}

fn main() {
    let s = fs::read_to_string("names.txt").unwrap();

    let alpha: Vec<_> = once('.').chain('a'..='z').collect();
    let n_alpha = alpha.len();
    let itos: HashMap<usize, char> = alpha.iter().copied().enumerate().collect();
    let stoi: HashMap<char, usize> = alpha.iter().enumerate().map(|(i, &x)| (x, i)).collect();

    let mut counts = Array2::<usize>::zeros((n_alpha, n_alpha));

    // let mut ws = s.lines();
    for w in s.lines() {
        let mut chars = wrap_dots(w);
        let mut prev = chars.next().unwrap();
        for ch in chars {
            counts[[stoi[&prev], stoi[&ch]]] += 1;
            prev = ch;
        }
    }
    let mut probs = counts.mapv(|x| x as f64);
    probs.rows_mut().into_iter().for_each(|mut row| {
        let sum = row.sum();
        row.mapv_inplace(|x| x / sum);
    });

    println!("{counts:?}");
    println!("{probs:?}");

    let dists: Vec<_> = counts
        .rows()
        .into_iter()
        .map(|row| WeightedIndex::new(row).unwrap())
        .collect();

    let mut rng = StdRng::seed_from_u64(42);
    let names: Vec<_> = (0..10).map(|_| generate_name(&mut rng, &dists, &itos)).collect();
    println!("{names:?}");
}
