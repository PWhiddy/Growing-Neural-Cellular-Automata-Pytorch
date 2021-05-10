
## Neural CA Dynamics Training  

A framework for training NCAs (based on https://distill.pub/2020/growing-ca/) which can model evolving systems with local interactions.

The system being modeled can be in the form of a pre-rendered video or live simulation.  
See ```run_video_ca_training.py``` for an example of training from video frames.
Running this script will create directory ```output/{datetime}/{checkpoints, rendered final states, system evolution}```  
The model weights can be exported to json using ```export_to_webgl.py {checkpoint path}``` for running in webgl like in the distill article.
The ```make_evo_video.sh``` and ```make_evo_gt_video.sh``` scripts can compile the rendered output states into easily digestible videos. 

Current limitations of video training:

- Video input w/h dimension should roughly be in the range 16-64 pixels.
- Dynamics in the video should be dominated by local interactions (no instantaneous communication between distant elements, next state should be possible to predict using only local information)
- System should mostly be deterministic. The degree of noise or random behavior the ca model can tolerate is untested at this time.
- All information needed to predict the next state should be given in the rgb pixels of the video. (Making the previous two frames available to the model could be very interesting as it broadens the options for bring in videos. This would require a small change to the training code and a medium change to the webgl demo to make this possible.)
- Boundary conditions should be considered. Fixed value and circular boundaries can be support with tiny code changes.
