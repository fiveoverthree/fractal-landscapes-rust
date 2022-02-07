use rand_distr::num_traits::Pow;
use rand_distr::{Normal, Distribution};
use rand::{thread_rng, Rng};
use linreg::linear_regression_of;
use std::fmt::Debug;
use std::io::Write;
use std::io::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use image::{ImageBuffer, Luma};
use emath;
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::prelude::*;

#[derive(Debug)]
struct Surface{
    sizemultiplier:usize,
    size:usize,
    surface:Vec<Vec<f64>>,
}

impl Surface {
    // Creates Surface from scratch. Corresponding Object has size 2^sizemultiplier + 1
    fn new(sizemultiplier: usize) -> Surface {
        let size = usize::pow(2, sizemultiplier as u32)+1;
        Surface{
            sizemultiplier,
            size,
            surface: vec![vec![0.0; size]; size],
        }
    }
    // Creates Surface from existing csv file with name (name)
    fn from_file(name: &str) -> Surface{
        fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
            where P: AsRef<Path>, {
            let file = File::open(filename)?;
            Ok(io::BufReader::new(file).lines())
        }
        let mut surface:Vec<Vec<f64>> = vec!(vec!());
        if let Ok(lines) = read_lines(name) {
            for (num0, line) in lines.enumerate() {
                if let Ok(row) = line {
                    for item in row.split(","){
                        surface[num0].push(item.parse().unwrap());
                    }
                    surface.push(vec![]);
                }
            }
            // pop the last element so we do not have an empty vector
            surface.pop();
        }
        let size = surface.len();
        let sizemultiplier = ((size - 1) as f64).sqrt() as usize;
        Surface{
            sizemultiplier,
            size,
            surface,
        }

    }
    // Write Surface to csv file. Path(name) needs to be provided
    fn write_to_file(&self, name: &str) {
        let mut f = File::create(name.to_owned() + ".csv").expect("Unable to create file");
        for row in &self.surface{
            let mut reihe = String::new();
            for cell in row{
                reihe.push_str(&cell.to_string());
                reihe.push(',');
            }
            // we do not want the last element to contain comma
            reihe.pop();
            writeln!(f, "{}", reihe).expect("Could not write to CSV file.")
        }
    }
    // write Surface to 16bit Greyscale PNG of name (name)
    fn write_to_image_file(&self, name: &str){
        let mut greyimage = ImageBuffer::new(self.size as u32, self.size as u32);
        for x in 0..((self.size) as u32){
            for y in 0..((self.size) as u32){
                // * 32768.0
                greyimage.put_pixel(x, y, Luma([(((self.surface[x as usize][y as usize] ) / self.size as f64) * 32768.0 )as u16]));
            }
        }
        greyimage.save_with_format("pict/".to_owned() + name + ".png", image::ImageFormat::Png).expect("Unable to write to PNG file.");
    }
    // set surface of a specific point to be x
    fn setsurface(&mut self, index: [usize; 2], value:f64) -> (){
        self.surface[index[0]][index[1]] = value;
    }
    // generate a Fractal terrain with a dimension d of the line-segments
    fn generate(&mut self, dim:f64) {
 
        fn calculate_midpoint(mean_height: f64, scale: f64, dim: f64) -> f64 {
            /*
            return midpoint of given scale
             */
            let sigma: f64 = 1.0;
            let normal = Normal::new(0.0, (sigma.powf(2.0)/(f64::powf(2.0,2.0*scale*dim))*(1.0-f64::powf(2.0, 2.0*dim-2.0)))*scale).unwrap();
            let var: f64 = normal.sample(&mut rand::thread_rng());
            mean_height+var

        }
        fn square_step(surface: &mut Vec<Vec<f64>>, split_multiplier: usize, shape_length: usize, dim: f64) -> (){
            for x in 0..split_multiplier {
                for y in 0..split_multiplier {
                    // coordinates
                    let x_min = x  * shape_length;
                    let x_max = (x+1)*shape_length;
                    let y_min = y * shape_length;
                    let y_max = (y + 1) * shape_length;
                    let x_mid = x_min + (shape_length / 2);
                    let y_mid = y_min + (shape_length / 2);
                    // neighbours
                    let n_w = surface[x_min][y_min];
                    let n_o = surface[x_min][y_max];
                    let s_w = surface[x_max][y_min];
                    let s_o = surface[x_max][y_max];
                    // calculate midpoint
                    surface[x_mid][y_mid] = calculate_midpoint((n_w + n_o + s_w + s_o) / 4.0, 1.0/ (split_multiplier as f64), dim);
                }
            }
        }
        fn diamond_step(surface: &mut Vec<Vec<f64>>, split_multiplier: usize, shape_length: usize, dim: f64) -> (){
            let half_size = shape_length / 2;
            let max = surface.len() as usize - 1;
            for x in 0..split_multiplier{
                for y in 0..split_multiplier{
                    // coordinates
                    let x_min = x * shape_length;
                    let x_max = (x + 1) * shape_length;
                    let y_min = y * shape_length;
                    let y_max = (y + 1) * shape_length;
                    let x_mid = x_min + (shape_length / 2);
                    let y_mid = y_min + (shape_length / 2);
                    // neighbours
                    let c = surface[x_mid][y_mid];
                    let n_w = surface[x_min][y_min];
                    let n_o = surface[x_min][y_max];
                    let s_w = surface[x_max][y_min];
                    let s_o = surface[x_max][y_max];

                    // Diamond steps
                    // top
                    // if at edge, only use 3 for calc
                    if surface[x_min][y_mid] == 0.0{
                        if x_min == 0{
                            surface[x_min][y_mid] = calculate_midpoint((c + n_w + n_o) / 3.0, 1.0/ (split_multiplier as f64), dim);
                        }
                        else {
                            surface[x_min][y_mid] = calculate_midpoint((c + n_w + n_o + surface[x_min - half_size][y_mid]) / 4.0, 1.0/ (split_multiplier as f64), dim);
                        }
                    }
                    // left
                    if surface[x_mid][y_min] == 0.0{
                        if y_min == 0{
                            surface[x_mid][y_min] = calculate_midpoint((c + n_w + s_w) / 3.0, 1.0/ (split_multiplier as f64), dim);
                        }
                        else{
                            surface[x_mid][y_min] = calculate_midpoint((c + n_w + s_w + surface[x_mid][y_min + half_size]) / 4.0, 1.0/ (split_multiplier as f64), dim);
                        }
                    }
                    // right
                    if surface[x_mid][y_max] == 0.0{
                        if y_max == max{
                            surface[x_mid][y_max] = calculate_midpoint((c + n_o + s_o) / 3.0, 1.0/ (split_multiplier as f64), dim);
                        }
                        else{
                            surface[x_mid][y_max] = calculate_midpoint((c + n_o + s_o + surface[x_mid][y_max + half_size]) / 4.0, 1.0/ (split_multiplier as f64), dim);
                        }
                    }
                    // bottom
                    if surface[x_max][y_mid] == 0.0{
                        if x_max == max{
                            surface[x_max][y_mid] = calculate_midpoint((c + s_w + s_o) / 3.0, 1.0/ (split_multiplier as f64), dim);
                        }
                        else{
                            surface[x_max][y_mid] = calculate_midpoint((c + n_o + s_o + surface[x_max + half_size][y_mid]) / 4.0, 1.0/ (split_multiplier as f64), dim);
                        }
                    }
                }
            }
        }
        fn initialize(surface: &mut Vec<Vec<f64>>, l:usize) -> (){
            surface[0][0] = 1.0;
            surface[0][l] = 1.0;
            surface[l][0] = 1.0;
            surface[l][l] = 1.0;
        }
        let mut l = self.surface.len() - 1;
        let mut grid_split:usize = 1;
        initialize(&mut self.surface, l);
        for _iteration in 0..self.sizemultiplier{
            square_step(&mut self.surface,grid_split, l, dim);
            diamond_step(&mut self.surface, grid_split, l, dim);
            l = (l / 2) as usize;
            grid_split *= 2;
        }

    }
    // calculate thermal erosion of t+1 with a talus angle of (talus_angle)
    fn thermal_erosion(&mut self, talus_angle:usize, erosion_factor: f64) -> (){
        let mut temp_surface = self.surface.clone();
        for num0 in 0..self.surface.len(){
            for num1 in 0..self.surface.len(){
                let mut elegible_neighbours:Vec<[usize; 2]> = vec!();
                // sum of all neighbour heights
                let mut neighbour_heights: f64 = 0.0;
                let mut lowest_neighbor: Option<f64> = None;
                for n in self.get_neighbours([num0, num1]){
                    // height of current neighbour
                    let n_h = self.surface[n[0]][n[1]];
                    //check if neighbour is eligible for further processing (talus angle)
                    // d = 1/self.size
                    if n_h < self.surface[num0][num1] && (self.surface[num0][num1] - n_h)*(self.size as f64) > ((talus_angle as f64).to_radians().atan()){
                        elegible_neighbours.push(n);
                        match lowest_neighbor{
                            Some(l_n) => {
                                if n_h < l_n{
                                    // if the height is smaller than all previous neighbours, make
                                    // it the lowest
                                    lowest_neighbor = Some(n_h);
                                }
                            }
                            None => {
                                // if no lowest neighbour exists to date, n is lowest one
                                lowest_neighbor = Some(n_h);
                            }
                        }
                        // add current height to neighbour heights
                        neighbour_heights += self.surface[num0][num1];
                    }
                }
                // if none of the neighbours is lower than current, check next cell
                if let None = lowest_neighbor {
                    continue
                }
                let lowest_neighbor = lowest_neighbor.unwrap();
                // maximum displaced soil
                // /8 to prevent disproportional growth
                let d_s = (self.surface[num0][num1] - lowest_neighbor)*erosion_factor/8.0;
                // for all neighbours that satisfy talus angle, we can add some sediment from original
                for neighbour in elegible_neighbours{
                    // reminder: neighbour heights must never be 0
                    temp_surface[neighbour[0]][neighbour[1]] += d_s * (temp_surface[neighbour[0]][neighbour[1]]/neighbour_heights);
                }
                // remove soil from original cell
                temp_surface[num0][num1] -= d_s;
            }
        }
        // update whole plane
        self.surface = temp_surface.clone();
    }
    // returns a list of indexes of all neighbours, including diagonal ones
    fn get_neighbours(&self, index: [usize; 2]) -> Vec<[usize; 2]>{
        let x = self.surface.len();
        let y = self.surface[0].len();
        let mut to_return:Vec<[usize; 2]> = vec!();
        // len starts at 1, so smaller than is used

        // non-diagonal
        // if we can make the x axis greater without going out of bounds, appen
        if index[0] + 1 < x {
            to_return.push([index[0] + 1, index[1]])
        }
        // x smaller w/o oub, append
        if index[0] >= 1{
            to_return.push([index[0] - 1, index[1]])
        }
        // y bigger w/o oub, append
        if index[1] + 1  < y{
            to_return.push([index[0], index[1] + 1])
        }
        // y smaller w/o oub, append
        if index[1] >= 1{
            to_return.push([index[0], index[1] - 1])
        }

        // diagonal
        // if we can make x and y greater, append
        if index[0] + 1 < x && index[1] + 1 < y{
            to_return.push([index[0] + 1, index[1] + 1])
        }
        // if we can make both smaller, append
        if index[0] >= 1 && index[1] >= 1{
            to_return.push([index[0] - 1, index[1] - 1])
        }
        // if wcm x ++ && y--, append
        if index[0] + 1 < x && index[1] >= 1{
            to_return.push([index[0] + 1, index[1] - 1])
        }
        // if wcm x-- && y++, append
        if index[0] >= 1 && index[1] + 1 < y{
            to_return.push([index[0] - 1, index[1] + 1])
        }
        to_return
    }
    /*
        Calculates the fractal dimension of the surface, starting from boxsize = surface.shape / start_coeff
        up until boxsize = to
        this is because at boxsize = surface.shape && boxsize = 1, the dimension is only dependent
        on the grid itself, with a dimension of 2.0, whereas in between these values lies
        the interesting part
    */
    fn fractal_dim(&self, start_coeff: usize, to: usize) -> f64{
        fn generate_bool_map(float_array: &Vec<Vec<f64>>, vertsplit: usize) -> Vec<Vec<Vec<bool>>>{
            // returns a 3d bool array in the shape z, x, y wether surface is present in that box
            let total_height: f64 = 2.;
            let mut result: Vec<Vec<Vec<bool>>> = float_array.iter().map(
                | row | {
                    row.iter().map( | height | {
                        (0..vertsplit).into_iter().map( | i | {
                            total_height / (vertsplit as f64) * ((i + 1) as f64) > *height &&
                                *height >= total_height / (vertsplit as f64) * (i as f64)
                        }).collect()
                    }).collect()
                }
            ).collect();
            result
        }
        fn boxcount(array: &Vec<Vec<f64>>, boxsize: usize) -> usize {

            let mut count = 0;
            let bool_map = generate_bool_map(array, array.len() / boxsize);
            // generate indices to index beforehand, making sure
            // to handle the edges, if it is not possible to
            // create a chunk with equal size
            let indices_z = (0..bool_map.len()).collect::<Vec<usize>>();
            let indices_z: Vec<&[usize]> = indices_z.chunks(boxsize).collect();
            let indices_x = (0..bool_map[0].len()).collect::<Vec<usize>>();
            let indices_x: Vec<&[usize]> = indices_x.chunks(boxsize).collect();
            let indices_y = (0..bool_map[0][0].len()).collect::<Vec<usize>>();
            let indices_y: Vec<&[usize]> = indices_y.chunks(boxsize).collect();
            for (chunk_iz, chunk_z) in indices_z.iter().enumerate() {
                for (chunk_ix, chunk_x) in indices_x.iter().enumerate() {
                    for (chunk_iy, chunk_y) in indices_y.iter().enumerate() {
                        // after selecting the chunk, now all the values
                        // in the chunk are being checked
                        let mut any_true = false;
                        for z in chunk_z.iter() {
                            if any_true {break}
                            for x in chunk_x.iter() {
                                if any_true {break}
                                for y in chunk_y.iter() {
                                    if bool_map[*z][*x][*y] {
                                        any_true = true;
                                        break
                                    }
                                }
                            }
                        }
                        if any_true { count += 1 }
                    }
                }
            }

            count
        }
        fn point_to_coordinates(number_of_boxes: usize, size_of_boxes: usize) -> (f64, f64){
            // returns the log of size and number, to be used for linear regression
            ((1.0/(size_of_boxes as f64)).ln(), ((number_of_boxes) as f64).ln())
        }
        let mut points:Vec<(f64, f64)> = vec!();
        // make surface of size 2^n, as this is more accurate
        let mut adjusted = self.surface.clone();
        for mut row in &mut adjusted{
            row.pop();
        }
        adjusted.pop();
        // actual calc
        let mut size = (self.surface.len() - 1) / start_coeff;
        while size >= to{
            points.push(point_to_coordinates(boxcount(&adjusted, size), size));
            size /= 2;
        }
        // calculating slope
        let coeff: Result<(f64, f64), linreg::Error> = linear_regression_of(&points);
        match coeff{
            Ok(res) => {
                return res.0
            }
            _ => {}
        }
        0.
    }
    fn hydraulic_erosion(&mut self, iterations: usize){
        let mut rng = thread_rng();
        let mut drop =  Droplet{
            pos: emath::Vec2::from([rng.gen_range(0.0..(self.size - 1)as f32), rng.gen_range(0.0..(self.size - 1) as f32)]),
            dir: emath::Vec2::from([0.0, 0.0]),
            vel: 1.0,
            water: 1.0,
            sediment: 0.0,
            inertia: 0.0, // high value = old dir taken into account more. range 0..1
            pminslope: 0.01,
            sCF: 2.0,
            erosion_radius: 2,
            deposition_speed: 0.05, // max 1, only when drop suddenly has more sediment than it can hold
            erosion_speed: 0.05, // max 1
            gravity: 10.0,
            evaporation_speed: 0.1,
    
        };
        drop.simulate(&mut self.surface, iterations, self.size)
    }
    // normalizes the array, so x,y,z have the same dimensions
    fn normalize_to_size(&mut self){
        let len = self.surface.len() as f64;
        for row in self.surface.iter_mut(){
            for cell in row.iter_mut(){
                *cell *= len;
            }
        }
    }
}
struct Droplet{
    pos: emath::Vec2,
    dir: emath::Vec2,
    vel: f32,
    water: f32,
    sediment: f32,
    inertia: f32,
    pminslope: f32,
    sCF: f32,
    erosion_radius: usize,
    deposition_speed: f32,
    erosion_speed: f32,
    gravity: f32,
    evaporation_speed: f32,   
}
impl Droplet{
    fn simulate(&mut self, surface: &mut Vec<Vec<f64>>, iterations: usize, size: usize) {
        // Calculates gradient vector for given position
        fn gradient(surface: &Vec<Vec<f64>>, position: &emath::Vec2) -> emath::Vec2{
            let cellCordsx = position.x.floor() as usize;
            let cellCordsy = position.y.floor() as usize;
            let x = position.x - cellCordsx as f32;
            let  y = position.y - cellCordsy as f32;
            let heightNW = surface[cellCordsx][cellCordsy] as f32;
            let heightNE = surface[cellCordsx + 1][cellCordsy] as f32;
            let heightSW = surface[cellCordsx][cellCordsy + 1] as f32;
            let heightSE = surface[cellCordsx + 1][cellCordsy + 1] as f32;
            //println!("neighbour_heigths: {}, {}, {}, {}", heightNW, heightNE, heightSW, heightSE);
            let gradientx = (heightNE - heightNW) * (1.0 - y) + (heightSE - heightSW) * y;
            let gradienty = (heightSW - heightNW) * (1.0 - x) + (heightSE - heightNE) * x;
            emath::Vec2::from([gradientx, gradienty])
        }
        // return sthe new direction in which the drop is flowing in
        fn dirnew(dir: &emath::Vec2, inertia: &f32, position: &emath::Vec2, surface: &Vec<Vec<f64>>) -> emath::Vec2{
            let x = (*dir * *inertia) - gradient(&surface, &position)*(1.0 - *inertia);
            if x.length() > 1.0{
                return emath::Vec2::new(x.x / f32::sqrt(x.x*x.x+x.y*x.y), x.y / f32::sqrt(x.x*x.x+x.y*x.y))
            } else {
                return x
            }
        }
        fn deposit(surface: &mut Vec<Vec<f64>>, pos: &emath::Vec2, amount: &f32){
            let cellCordsx = pos.x.floor() as usize;
            let cellCordsy = pos.y.floor() as usize;
            let x = pos.x - cellCordsx as f32;
            let y = pos.y - cellCordsy as f32;
            //println!("x, y: {}, {}", x, y);
            // NW
            surface[cellCordsx][cellCordsy] += (amount * (1.0-x) * (1.0-y)) as f64;
            // NE
            surface[cellCordsx + 1][cellCordsy] += (amount * x * (1.0 - y)) as f64;
            // SW
            surface[cellCordsx][cellCordsy + 1] += (amount * (1.0 - x) * y) as f64;
            // SE
            surface[cellCordsx + 1][cellCordsy + 1] += (amount * x* y) as f64;
        }
        fn erode(surface: &mut Vec<Vec<f64>>, pos: &emath::Vec2, amount: &f32, erosion_radius: &usize){
            // calculate maxima to determine weights
            // TODO VERY INEFFICIENT
            let mut s:f64 = 0.0;
            for (num0, row) in surface.iter().enumerate(){
                for (num1, cell) in row.iter().enumerate(){
                    s += f64::max(0.0, (*erosion_radius as f64) - (emath::vec2(num0 as f32, num1 as f32) - *pos).length() as f64);
                }
            }
            // erode
            for (num0, row) in surface.clone().iter().enumerate(){
                for (num1, _cell) in row.iter().enumerate(){
                    surface[num0][num1] -= (f64::max(0.0, *erosion_radius as f64 - ((emath::vec2(num0 as f32, num1 as f32) - *pos)).length() as f64)/s)*(*amount as f64);
                }
            }
        }
        fn mean_height(surface: &Vec<Vec<f64>>, position: &emath::Vec2) -> f32{
            let cellCordsx = position.x.floor() as usize;
            let cellCordsy = position.y.floor() as usize;
            let x = position.x - cellCordsx as f32;
            let  y = position.y - cellCordsy as f32;
            let heightNW = surface[cellCordsx][cellCordsy] as f32;
            let heightNE = surface[cellCordsx + 1][cellCordsy] as f32;
            let heightSW = surface[cellCordsx][cellCordsy + 1] as f32;
            let heightSE = surface[cellCordsx + 1][cellCordsy + 1] as f32;
            // weighted arithmetic mean: Sum(weights*heights)/sum(weights)
            heightNW*((1.0-x) * (1.0-y)) 
            + heightNE*( x * (1.0-y)) 
            + heightSE*(x*y)
            + heightSW*((1.0-x) * y)
        }
        fn sedimentCapacity(h_diff: &f32, pminslope: &f32, vel: &f32, water: &f32, sedimentCapacityFactor: &f32) -> f64{
            (f32::max(-h_diff, *pminslope) * vel * water * sedimentCapacityFactor) as f64
        }
        for x in 0..iterations{
            let h_old = mean_height(&surface, &self.pos);
            let pos_old = emath::vec2(self.pos.x.clone(), self.pos.y.clone());
            // update position
            self.dir = dirnew(&self.dir, &self.inertia, &self.pos, &surface);
            self.pos += self.dir;
           
            // drop droplet if it goes out of bounds
            if self.pos.x < 0.0 || self.pos.x >= (size as f32 - 1.0) || self.pos.y < 0.0 || self.pos.y >= (size as f32 - 1.0){
                break;
            }
            
            // exit if not moving
            if self.dir.length() == 0.0{
                println!("exiting");
                break;
            }
            

            let h_new = mean_height(&surface, &self.pos);
            let deltaheight = h_new-h_old;
            let sedCap = sedimentCapacity(&deltaheight, &self.pminslope, &self.vel, &self.water, &self.sCF);
            // we deposit if moving uphill or self.sediment > sedCap
            // moving uphill, deposit at old pos to close gap
            if deltaheight > 0.0{
                let to_deposit = f32::min(deltaheight, self.sediment);
                self.sediment -= to_deposit;
                deposit(surface, &pos_old, &to_deposit);
            } else {
                if self.sediment > sedCap as f32{
                    // drop has more sediment than it can hold -> depositing
                    let to_deposit = (self.sediment - sedCap as f32) * self.deposition_speed;
                    self.sediment -= to_deposit;
                    deposit(surface, &pos_old, &to_deposit);
                } else {
                    // if drop has less sediment than it is allowed to, erode from the map
                    let to_erode = f32::min((sedCap as f32 - self.sediment)*self.erosion_speed, -deltaheight);
                    self.sediment += to_erode;
                    erode(surface, &pos_old, &to_erode, &self.erosion_radius);
                }
            }
            // calculate new speed
            // CHANGED
            self.vel = f32::sqrt(f32::max(self.vel.powf(2.0) - (deltaheight*self.gravity), 0.0));
            // calculate new water level
            self.water *= 1.0-self.evaporation_speed;
        }
    }
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

fn main() {
    // variables
    let sizemultiplier = 9;
    let mut dH = 1.0;
    let erosion_steps = 1000;
    // list of all steps (dH) we are going to calculate
    let mut dim_steps:Vec<Vec<Surface>> = (0..=9).map(|_x| {
        (0..=9).map(|_x| {Surface::new(sizemultiplier)}).collect()
    }).collect();
    // generating all the surfaces
    for mut testrow in&mut dim_steps{
        testrow.par_iter_mut().for_each(|sf|{
            sf.generate(dH)           
        });
        dH += 0.1;
    }
    // fractal dimension calculation
    dim_steps.iter().for_each(|row|{
        let dims_row = row.par_iter().map(|s|s.fractal_dim(4, 2)).collect();
        // append data to csv file
        append_results_to_file(dims_row, "results");
    });
    // thermal erosion
    for _ in 0..=erosion_steps{ 
        append_results_to_file(Vec::new(), "results");
        // erode one time
        dim_steps.iter_mut().for_each(|row|{
            row.par_iter_mut().for_each(|s|s.thermal_erosion(45, 2.0));
        });
        // fractal dimension calculation
         dim_steps.iter().for_each(|row|{
             let dims_row = row.par_iter().map(|s|s.fractal_dim(4, 2)).collect();
             // append data to csv file
             append_results_to_file(dims_row, "results");
         });
  
    }
}
