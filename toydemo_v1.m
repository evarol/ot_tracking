clear
clc
close all



vec=@(x)(x(:));

%%fake data generation
dim = 100; %square image size
T=20; %time points
noise=10; %noise level
radius=10; %ball size

[X,Y]=meshgrid((1:dim),(1:dim));
coor = [X(:) Y(:)];
data = noise*rand(size(coor,1),T);
for t=1:T
    data(sqrt(sum((coor - [20 10+t/1]).^2,2))<=radius,t)=8+noise*rand(size(data(sqrt(sum((coor - [20 10+t/1]).^2,2))<=radius,t)));
    data(sqrt(sum((coor - [40 10+t/0.6]).^2,2))<=radius,t)=8+noise*rand(size(data(sqrt(sum((coor - [40 10+t/0.6]).^2,2))<=radius,t)));
    data(sqrt(sum((coor - [60 10+t/0.35]).^2,2))<=radius,t)=8+noise*rand(size(data(sqrt(sum((coor - [60 10+t/0.35]).^2,2))<=radius,t)));
    data(sqrt(sum((coor - [80 10+t/0.25]).^2,2))<=radius,t)=8+noise*rand(size(data(sqrt(sum((coor - [80 10+t/0.25]).^2,2))<=radius,t)));
end


V=reshape(data,[dim dim T]); %data reshape to matrix


for t=1:size(V,3)
    tmp(:,:,t)=imresize(V(:,:,t),0.35); %downsample image for OT computation
end


%% Duplicate and paste video
V=tmp; clear tmp
nsink=100;
for t=1:size(V,3)
    V(:,:,T+t)=V(:,:,T-t+1);
end
sequential=0;


%% choose point(s) to track
% trackpoint = [4 7;4 21;4 29];
[x,y]=meshgrid(1:5:35,1:5:35);
trackpoint=[x(:) y(:)];


%% Optimal transport from frame to frame -ENCODING STEP
for t=1:size(V,3)-1
    if sequential~=1 %OT from frame 1 to all frames
    [H{t},Yhat{t}]=optimal_transport(V(:,:,1),V(:,:,t+1),0.05,100); %% OT parameters
    elseif sequential==1 %OT from frame t to t+1
        [H{t},Yhat{t}]=optimal_transport(V(:,:,t),V(:,:,t+1),0.05,100);
    end
    
    
H{t}=H{t}/sum(vec(H{t}));
    ['Encoding ' num2str(t) '/' num2str(T)]
end

Vhat(:,:,1)=V(:,:,1)./sum(vec(V(:,:,1)));

%% Optimal transport -DECODING STEP
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




%% Visualization
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
if sequential==0
    for t=size(V,3)-1:-1:1
        c=H{t}'*vec(Zhat(:,:,end,T));c=c/sum(c);
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
Zhat=tmpzhat;clear zhat
That=tmpthat;clear tmpthat
V(:,:,:,2)=V(:,:,:,1);
V(:,:,:,3)=V(:,:,:,1);
Vhat(:,:,:,2)=Vhat(:,:,:,1);
Vhat(:,:,:,3)=Vhat(:,:,:,1);
% Zhat(:,:,:,2:3)=0;
% imshow3d(cat(2,V./max(V(:)),Vhat./max(Vhat(:)),Zhat./max(Zhat(:)),That./max(That(:))))
 imshow3d(cat(2,V(:,:,:,1)./max(V(:)),Vhat(:,:,:,1)./max(Vhat(:)),max(Zhat,[],4)./max(Zhat(:)),max(That,[],4)./max(That(:))))