 % load /Users/erdem/Dropbox/Projects/worm/gcamp_traces/animal_018_tail/run_tail_407.mat
% 
% X=reshape(data,[size(data,1)*size(data,2) size(data,3)]);
clearvars -except green
close all
clc
vec=@(x)(x(:));
if ~exist('green')
%load('/Users/erdem/Dropbox/Projects/worm/hillman_video.mat','green');
load('/Users/cmcgrory/paninski_lab/worm/data/hillman_video.mat');
green = permute(green,[2 1 3 4]);
end

T=size(green,4);

for i=1:size(green,4)
    V(:,:,i)=sum(green(:,:,:,i),3);
end
V=double(V);

for t=1:size(V,3)
    tmp(:,:,t)=imresize(V(:,:,t),0.24);
end
V=tmp; clear tmp
nsink=100;
% for t=1:size(V,3)
%     V(:,:,T+t)=V(:,:,T-t+1);
% end
sequential=0;

[x,y]=meshgrid(1:5:28,1:5:159);
trackpoint=[x(:) y(:)];
for t=1:size(V,3)-1
    if sequential~=1
    [H{t},Yhat{t}]=optimal_transport(V(:,:,1),V(:,:,t+1),5,2);
    elseif sequential==1
        [H{t},Yhat{t}]=optimal_transport(V(:,:,t),V(:,:,t+1),5,2);
    end
%     for s=1:nsink
%         H{t}=H{t}./sum(H{t},2);
%         H{t}=H{t}./sum(H{t},1);
%     end
H{t}=H{t}/sum(vec(H{t}));
% H{t}=H{t}./sqrt(sum(H{t}.^2,2));
%     H{t}=munkres(H{t});
% [I,J]=max(H{t},[],2);

    % hungarian on H maybe?
    ['Encoding ' num2str(t) '/' num2str(T)]
end

Vhat(:,:,1)=V(:,:,1)./sum(vec(V(:,:,1)));
if sequential==1
for t=1:length(Yhat)
    a=vec(Yhat{t});a=a./sum(a);
    mapto=t;
    while mapto~=1
        a=H{mapto-1}*a;
        mapto=mapto-1;
        a=a./sum(a);
    end
    Vhat(:,:,t+1)=reshape(a,size(Yhat{1}));
    ['Decoding ' num2str(t) '/' num2str(T)]
end
elseif sequential~=1
    for t=1:length(Yhat)
    a=vec(Yhat{t});a=a./sum(a);
    Vhat(:,:,t+1)=reshape(a,size(Yhat{1}));
    end
end

Zhat=zeros([size(V) size(trackpoint,1)]);
That=zeros([size(V) size(trackpoint,1)]);
for T=1:size(trackpoint,1)
Zhat(trackpoint(T,1),trackpoint(T,2),end,T)=1;
That(trackpoint(T,1),trackpoint(T,2),end,T)=1;
if sequential==1
    for t=size(V,3)-1:-1:1
        c=vec(Zhat(:,:,t+1,T));c=c/sum(c);
        c=H{t}*c;
        c=c/sum(c);
        Zhat(:,:,t,T)=reshape(c,size(Yhat{1}));
        [~,idx]=max(c);
        tmp=zeros(size(c));
        tmp(idx)=1;
        That(:,:,t,T)=reshape(tmp,size(Yhat{1}));
    end
end
clear tmp
% Vhat=log(Vhat./max(Vhat(:)));
% V=log(V./max(V(:)));
% Vhat=Vhat-min(Vhat(:));
% V=V-min(V(:));

for i=1:size(Zhat,3)
    Zhat(:,:,i,T)=Zhat(:,:,i,T)/max(vec(Zhat(:,:,i,T)));
    That(:,:,i,T)=That(:,:,i,T)/max(vec(That(:,:,i,T)));
end
end



for t=1:size(V,3)
    tmp(:,:,t)=V(:,:,end-t+1);
    tmphat(:,:,t)=Vhat(:,:,end-t+1);
    tmpzhat(:,:,t,:)=Zhat(:,:,end-t+1,:);
    tmpthat(:,:,t,:)=That(:,:,end-t+1,:);
end
V=tmp;clear tmp
Vhat=tmphat;clear tmphat
Zhat=tmpzhat;clear tmpzhat
That=tmpthat;clear tmpthat
V(:,:,:,2)=V(:,:,:,1);
V(:,:,:,3)=V(:,:,:,1);
Vhat(:,:,:,2)=Vhat(:,:,:,1);
Vhat(:,:,:,3)=Vhat(:,:,:,1);
% Zhat(:,:,:,2:3)=0;
imshow3d(cat(2,V./max(V(:)),Vhat./max(Vhat(:)),Zhat./max(Zhat(:)),That./max(That(:))))