use std::{collections::HashMap, fs, io, iter};

use ndarray::{Array2, Axis};
use rand::{
    RngExt, SeedableRng,
    distr::{Distribution, weighted::WeightedIndex},
    rngs::StdRng,
};

// fn names() -> Vec<String> {
// }

fn main() -> Result<(), io::Error> {
    let mut counts: HashMap<(String, String), usize> = HashMap::new();
    let ws: Vec<String> = fs::read_to_string("names.txt")?
        .lines()
        .map(|s| str::to_owned(s))
        .collect();

    let mut tly: Vec<_> = counts.iter().collect();
    tly.sort_unstable_by(|a, b| b.1.cmp(a.1));

    for (bigram, count) in tly.iter().take(20) {
        println!("{bigram:?}: {count}");
    }

    let mut stoi: HashMap<String, usize> = HashMap::new();
    let crng = 'a'..='z';
    // println!("{:?}", crng.clone().collect::<Vec<_>>());

    stoi.insert("<S>".to_string(), 0);
    for (i, c) in crng.enumerate() {
        stoi.insert(c.to_string(), i + 1);
    }
    stoi.insert("<E>".to_string(), 27);

    println!("{:?}", stoi);
    let itos: HashMap<usize, String> = stoi.iter().map(|(a, &b)| (b, a.clone())).collect();
    let mut cnts = Array2::<usize>::zeros((28, 28));
    // for
    for w in ws {
        let tokens: Vec<_> = iter::once("<S>".to_string())
            .chain(w.chars().map(|s| s.to_string()))
            .chain(iter::once("<E>".to_string()))
            .collect();
        for t in tokens.windows(2) {
            let a = t[0].clone();
            let b = t[1].clone();

            cnts[[stoi[&a], stoi[&b]]] += 1;
        }
    }

    let mut rng = StdRng::seed_from_u64(43);
    let mut xs2: [f64; 3] = rng.random();

    let mut xs: Vec<_> = (0..3).map(|_| rng.random()).collect();
    let tot: f64 = xs.iter().sum();
    xs = xs.into_iter().map(|x| x / tot).collect();
    println!("{xs:?}");

    let dist = WeightedIndex::new(&xs).unwrap();
    let res: Vec<_> = (0..=100).map(|_| dist.sample(&mut rng)).collect();
    println!("{res:?}");

    // counts.
    let mut P = cnts.mapv(|x| x as f64);
    let row_sums = P.sum_axis(Axis(1)).insert_axis(Axis(1));
    P /= &row_sums;
    println!("{P:?}");
    Ok(())
}
