use std::{
    cmp::{self, Reverse},
    collections::HashMap,
    fs, iter,
};

use ndarray::Array2;
use rand::{
    SeedableRng,
    distr::{Distribution, weighted::WeightedIndex},
    rngs::StdRng,
};
use rand_distr::{Normal, StandardNormal};

fn main() {
    let s = fs::read_to_string("names.txt").unwrap();
    let ns: Vec<_> = s.lines().collect();
    let mut counts: HashMap<(char, char), usize> = HashMap::new();

    let mut stoi: HashMap<char, usize> = HashMap::new();
    stoi.insert('.', 0);
    ('a'..='z').enumerate().for_each(|(idx, ch)| {
        stoi.insert(ch, idx + 1);
    });
    println!("{:?}", stoi);
    let mut itos: HashMap<usize, char> = stoi.iter().map(|(&k, &v)| (v, k)).collect();
    // [i,j]'s value is the number of occurences from char i to char j
    let mut C = Array2::zeros((27, 27));
    C[[0, 0]] = 1;
    // iter over the names, prepend and append "." to the name
    // iter over windows(2) of the name
    // increment that tuple of characters in a hasmap
    for n in &ns {
        let mut n_str = n.to_string();
        n_str.insert(0, '.');
        n_str.insert(n_str.len(), '.');
        for w in n_str.chars().collect::<Vec<_>>().windows(2) {
            let tup = (stoi[&w[0]], stoi[&w[1]]);
            C[tup] += 1;
        }
    }
    println!("{:?}", C);
    let mut P = C.mapv(|a| a as f64);
    for mut row in P.rows_mut() {
        let sum = row.sum();
        row.mapv_inplace(|x| x / sum);
    }
    println!("{:?}", P);
    println!(
        "row sums: {:?}",
        P.rows().into_iter().map(|r| r.sum()).collect::<Vec<_>>()
    );

    let mut rnd = StdRng::seed_from_u64(43);

    let dists: Vec<_> = C
        .rows()
        .into_iter()
        .map(|r| WeightedIndex::new(r).unwrap())
        .collect();
    for _ in 0..20 {
        let mut cur = dists[0].sample(&mut rnd);
        loop {
            if cur == 0 {
                break println!();
            }
            cur = dists[cur].sample(&mut rnd);
            print!("{}", itos[&cur]);
        }
    }
    // rand::randn?
    // Normal
    let xn: f64 = StandardNormal.sample(&mut rnd);
    for n in &ns[0..3] {
        let mut n_str = n.to_string();
        n_str.insert(0, '.');
        n_str.insert(n_str.len(), '.');
        for w in n_str.chars().collect::<Vec<_>>().windows(2) {
            let (l, r) = (stoi[&w[0]], stoi[&w[1]]);
            println!("{}{}: {}", &w[0], &w[1], P[[l, r]])
        }
    }

    let a = Array2::<f64>::from_shape_fn((10, 10), |_| StandardNormal.sample(&mut rnd));
    let x = Array2::<f64>::from_shape_fn((10, 1), |_| StandardNormal.sample(&mut rnd));


    let ax = a*x;
    println!("{ax:?}");
}
