
## Neural CA Dynamics Training  

A framework for training NCAs (based on https://distill.pub/2020/growing-ca/) which can model the dynamics of evolving systems.

The system being modeled can be in the form of a pre-rendered video or live simulation.  
See ```run_video_ca_training.py``` for an example of training from video frames.
Running this script will create directory ```output/{datetime}/{checkpoints, rendered final states, system evolution}```  
The model weights can be exported to json using ```export_to_webgl.py {checkpoint path}``` for running in webgl like in the distill article.
The ```make_evo_video.sh``` and ```make_evo_gt_video.sh``` scripts can compile the rendered output states into easily digestable videos. 