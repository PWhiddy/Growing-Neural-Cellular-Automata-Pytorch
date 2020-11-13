rm ./sim_output/*.png
python run_sim.py
ffmpeg -y -framerate 60 -i ./sim_output/particles_step_%06d.png -pix_fmt yuv420p sim_output_particles.mp4
