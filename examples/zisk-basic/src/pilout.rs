use common::WitnessPilOut;

pub fn get_fibonacci_vadcop_pilout() -> WitnessPilOut {
    let pilout = WitnessPilOut::new("FibonacciVadcopPilOut", 2, b"fibonacci-vadcop-hash".to_vec());

    // let mut air_group = AirGroup::new("AirGroup_1");
    // air_group.add_air(BasicAir::new("FibonacciSquare", 8));
    // air_group.add_air(BasicAir::new("Module", 8));

    // pilout.add_air_group(air_group);

    pilout
}