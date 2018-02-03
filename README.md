# yolo_with_opencv

******************************************************************************************************

    SUPPORTED NETWORKS: yolo.cfg, yolo-voc.cfg, tiny-yolo.cfg, tiny-yolo-voc.cfg - can be downloaded: 
    https://drive.google.com/drive/folders/0BwRgzHpNbsWBN3JtSjBocng5YW8
    
    UNSUPPORTED NETWORKS: yolo9000.cfg
    
******************************************************************************************************


In order to execute correctly the code, modify the following lines.	

    #132 const float confidenceThreshold = <set-the-desired-value>;
    #133 const string namesPath = "[PATH-TO-DARKNET]/data/coco.names"
    #134 const string source = "<path-to-the-image-or-video-to-process>";
    #135 const string tinyModelBinary = "<path-to-'tiny-yolo.weights'-file>";
    #136 const string tinyModelConfiguration = "<path-to-'tiny-yolo.cfg'-file>";
    #137 const string yoloModelBinary = "<path-to-'yolo.weights'-file>";
    #138 const string yoloModelConfiguration = "<path-to-'yolo.cfg'-file>";

    
Change the following Rect objects (i.e. impact zones) w.r.t. the video's resolution.
   
    #156 const Rect proximityZone = Rect(<set-properly>);
    #157 const Rect movementZone = Rect(<set-properly>); 

    
