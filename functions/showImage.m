function [ img_handle ] = showImage( img_vector )
%SHOWIMAGE Reshape and display a Vectorized Image

s = length(img_vector);


if s == 784;
    % MNIST Image
    img = reshape(img_vector,[28,28]);
elseif s == 1200
    % ORL Image
    img = reshape(img_vector,[40,30]);
end

img_handle = imshow(img);

end

