
fn old_main() {
    // let a = 3.5;
    // let b = -2.;
    // let c = a * b;
    // let d = c + a; // ab + a = a(b + 1)
    // dd/dc = 1

    // dd/da = b + 1 = -1
    // dd/db = a = 3.5
    // d = -7 + 3.5 = -3.5

    let a = Val::new(3.5);
    let b = Val::new(-2.);
    let c = &a * &b;
    let d = &c + &a;
    let o = d.tanh();
    // o.set_grad(1.0);
    // o = a + b
    // do/da = 1
    // do/db = 1
    // o.local_backwards();
    // c.local_backwards(); // dd/da = -2 WRONG. dd/db = 3.5

    o.backward();

    let graph_path = dot::render_graph(&o, "graph").unwrap();
    println!("{graph_path:?}");

    // local_back on an Op::Add
    // c = a + b
    // d = 5*c
    // 5(a+b)
    // 5a + 5b
    // dd/da = 5 = dd/db
    // dd/dc = 5
    // dc/d

    // local_back on an Op::Mul
    // c = ab
    // d = 5c
    // d = 5ab
    // dd/da = 5b
    // dd/db = 5a

    // dd/dc = 5
    // dc/da = b
    // dd/da = 5b

    println!("o: {o:?}");

    println!("{}", (0..10).len());
}
