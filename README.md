# fractal-landscapes-rust
Generation and erosion of fractal landscapes
This project is licensed under the GPL Version 3 License and was created by Jonas Kriegl.

# Usage
In the the main.rs-file there are several examples of how to use the code. First, a new 'Surface' struct needs to be constructed using either 'Surface:new(sizemultiplier)' or 'Surface::from_file(file)' which loads a surface from a .csv file. An example surface can be found under 'test.csv'. Afterwards, a fractal landscape can be generated using 'Surface::generate(H)', where H is used to pinpoint the fractal dimension of the resulting object. This factor is inverse and is to be picked between 0 and 1. 1 is therefore resulting in lower fractal dimensions, while 0 has the highest. Afterwards, several kinds of erosion can be applied: Thermal erosion and hydraulic erosion (see below for further reference). Then, the fractal dimension can be calculated using 'Surface::fractal_dim(start_coeff, to)', where 'start_coeff' is a factor to determine the first boxsize to use and 'to' the last one. This is done because the first and last boxsizes always have a fractal dimension of exactly two. As a last step, the calculated surfaces can be saved as a csv or a 16 bit PNG heightmap.

# References
Benoit Mandelbrot: The fractal geometry of nature\\
https://ranmantaru.com/blog/2011/10/08/water-erosion-on-heightmap-terrain/ \\
https://www.wanderinformatiker.at/unipages/FractalTerrains/Fraktale_Terrainerzeugung.pdf \\
https://www.firespark.de/resources/downloads/implementation%20of%20a%20methode%20for%20hydraulic%20erosion.pdf \\
http://hpcg.purdue.edu/bbenes/papers/Benes02WSCG.pdf \\
https://old.cescg.org/CESCG-2011/papers/TUBudapest-Jako-Balazs.pdf \\
