% clear all
% clc
% close all

vidfile = VideoWriter('ot_track_syn_2d.mp4','MPEG-4');

open(vidfile)
M=track_vid; %define M as the stack of images you want to make a movie out of
for j=1:size(M,3)
    im=imresize(squeeze(M(:,:,j,:)),5,'nearest'); %upsampling for larger video size
    im=im-min(im(:)); %make frames between 0 and 1
    im=im./max(im(:));
    for t=1:4 % make 4 FPS
        writeVideo(vidfile,im);
    end
end
close(vidfile)