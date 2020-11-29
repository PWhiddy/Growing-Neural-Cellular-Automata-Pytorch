rm ./sim_output/*.png
python run_sim.py
ffmpeg -y -loglevel warning -framerate 60 -i ./sim_output/particles_step_%06d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p sim_output_particles.mp4
