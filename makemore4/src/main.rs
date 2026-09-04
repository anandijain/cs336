use std::{
    collections::HashMap,
    fs,
    iter::{self, once},
};

use ndarray::Array2;

fn wrap_dots<'a>(s: &'a str) -> impl Iterator<Item = char> + 'a {
    iter::once('.').chain(s.chars()).chain(once('.'))
}

fn word_probability(probs: &Array2<f64>, stoi: &HashMap<char, usize>, s: &str) -> f64 {
    let mut chrs = wrap_dots(s);
    let mut prev = chrs.next().unwrap();
    let mut p = 1.0;
    chrs.for_each(|c| {
        p *= probs[[stoi[&prev], stoi[&c]]];
        prev = c;
    });
    p
}

fn main() {
    let s = fs::read_to_string("names.txt").unwrap();
    let ls = s.lines();
    let alpha: Vec<_> = iter::once('.').chain('a'..='z').collect();
    let stoi: HashMap<char, usize> = alpha.iter().enumerate().map(|(i, &x)| (x, i)).collect();
    let mut counts = Array2::<usize>::zeros((alpha.len(), alpha.len()));
    for l in ls {
        let mut chars = wrap_dots(l);
        let mut prev = chars.next().unwrap();
        for ch in chars {
            counts[[stoi[&prev], stoi[&ch]]] += 1;
            prev = ch;
        }
    }
    println!("{counts:?}");

    let mut probs = counts.mapv(|x| x as f64);
    probs.rows_mut().into_iter().for_each(|mut row| {
        let sum = row.sum();
        row.mapv_inplace(|x| x / sum)
    });
    println!("{probs:?}");
    let name = "anand";
    println!("prob of word {name}: {}", word_probability(&probs, &stoi, &name));
}
