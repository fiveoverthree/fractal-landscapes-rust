mod lib;
use std::fs::OpenOptions;

use fractal_landscapes_rust::*;
use rayon::prelude::*;
use std::io::prelude::*;
/// REMINDER: H is inverse!!

fn main(){
    //different_H_and_thermal_erosion();
    println!("Finished h");
    //different_thermal_erosion_talus();
    println!("Finished talus");
    hydraulics();
}

fn append_results_to_file(res:Vec<f64>, filename:&str) -> (){
    let mut line = String::new();
    for item in res{
        line.push_str(&item.to_string());
        line.push(',');
    }
    line.pop();
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open(filename.to_owned() + ".csv")
        .unwrap();

    if let Err(e) = writeln!(file, "{}", line) {
        eprintln!("Couldn't write to file: {}", e);
    }
}

fn hydraulics(){
     // variables
    let sizemultiplier = 9;
    let h = 0.2;
    let erosion_steps = 10000;
    // list of all erosion radii that are goint to get calculated (radii 1,2,4,8)
    let mut radii_surfaces:Vec<Vec<Surface>> = (0..=3).map(|_x| {
        (0..=15).map(|_x| {Surface::new(sizemultiplier)}).collect()
    }).collect();
    // generating all the surfaces
    for mut row in &mut radii_surfaces{
        row.par_iter_mut().for_each(|sf|{
            sf.generate(h)           
        });
    }
    // fractal dimension calculation
    radii_surfaces.iter().for_each(|row|{
        let dims_row = row.par_iter().map(|s|s.fractal_dim(4, 2)).collect();
        // append data to csv file
        append_results_to_file(dims_row, "hydraulic");
    });
    println!("Finished initialization, going to erode now...");
    for step in 0..=erosion_steps{
        append_results_to_file(Vec::new(), "hydraulic");
        // erode 500 times (because this is very slow)
        for substep in 0..100{
            radii_surfaces.iter_mut().enumerate().for_each(|(i, row)|{
                row.par_iter_mut().for_each(|s|s.hydraulic_erosion(30, 0.01, usize::pow(2, i as u32)));
            });
            println!("Substep done {}", substep)
        }
        println!("Finished erosion for step {}, calculating fractal dimension...", step);
        // fractal dimension calculation
        radii_surfaces.iter().for_each(|row|{
            let dims_row = row.par_iter().map(|s|s.fractal_dim(4, 2)).collect();
            // append data to csv file
            append_results_to_file(dims_row, "hydraulic");
        });
        println!("{}", step)
    }
}

/// calculates different thermal erosion angles at a fixed dimension. Angles are from 10-80°
fn different_thermal_erosion_talus(){
    // variables
    let sizemultiplier = 9;
    let h = 0.2;
    let erosion_steps = 100;
    // list of all steps (dH) we are going to calculate
    let mut dim_steps:Vec<Vec<Surface>> = (0..=7).map(|_x| {
        (0..=9).map(|_x| {Surface::new(sizemultiplier)}).collect()
    }).collect();
    // generating all the surfaces
    for mut testrow in&mut dim_steps{
        testrow.par_iter_mut().for_each(|sf|{
            sf.generate(h)           
        });
    }
    // fractal dimension calculation
    dim_steps.iter().for_each(|row|{
        let dims_row = row.par_iter().map(|s|s.fractal_dim(4, 2)).collect();
        // append data to csv file
        append_results_to_file(dims_row, "different_talus");
    });
    // thermal erosion
    for _ in 0..=erosion_steps{ 
        append_results_to_file(Vec::new(), "different_talus");
        // erode one time
        dim_steps.iter_mut().enumerate().for_each(|(i, row)|{
            row.par_iter_mut().for_each(|s|s.thermal_erosion((i+1)*10, 1.0));
        });
        // fractal dimension calculation
        dim_steps.iter().for_each(|row|{
            let dims_row = row.par_iter().map(|s|s.fractal_dim(4, 2)).collect();
            // append data to csv file
            append_results_to_file(dims_row, "different_talus");
        });
  
    }

}

/// calculates dimension at different H values & their thermal erosion
fn different_H_and_thermal_erosion() {
    // variables
    let sizemultiplier = 9;
    let mut dH = 1.0;
    let erosion_steps = 200;
    // list of all steps (dH) we are going to calculate
    let mut dim_steps:Vec<Vec<Surface>> = (0..=9).map(|_x| {
        (0..=9).map(|_x| {Surface::new(sizemultiplier)}).collect()
    }).collect();
    // generating all the surfaces
    for mut testrow in&mut dim_steps{
        testrow.par_iter_mut().for_each(|sf|{
            sf.generate(dH)           
        });
        dH -= 0.1;
    }
    // fractal dimension calculation
    dim_steps.iter().for_each(|row|{
        let dims_row = row.par_iter().map(|s|s.fractal_dim(4, 2)).collect();
        // append data to csv file
        append_results_to_file(dims_row, "different-h-and-thermal");
    });
    // thermal erosion
    for _ in 0..=erosion_steps{ 
        append_results_to_file(Vec::new(), "different-h-and-thermal");
        // erode one time
        dim_steps.iter_mut().for_each(|row|{
            row.par_iter_mut().for_each(|s|s.thermal_erosion(45, 1.0));
        });
        // fractal dimension calculation
         dim_steps.iter().for_each(|row|{
             let dims_row = row.par_iter().map(|s|s.fractal_dim(4, 2)).collect();
             // append data to csv file
             append_results_to_file(dims_row, "different-h-and-thermal");
         });
  
    }
}
