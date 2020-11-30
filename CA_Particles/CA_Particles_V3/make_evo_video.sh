ffmpeg -y -loglevel warning -framerate 30 -i ./evolution_output/evo_step_%06d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p evolution.mp4
