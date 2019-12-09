clc;
clear all;
close all;
out = read_ecl('MASTER0.UNRST');
Pressure=out.PRESSURE;
saturation=out.SWAT;

save ( 'Pressure_ensemble.out', 'Pressure' , '-ascii')
save ( 'Saturation_ensemble.out', 'saturation' , '-ascii')
