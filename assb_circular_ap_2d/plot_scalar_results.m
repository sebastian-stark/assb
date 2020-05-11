clear;

r1_ = dlmread('results_global_ref_0_inc_20/independent_scalars.dat');
r2_ = dlmread('results_global_ref_0_inc_40/independent_scalars.dat');
r3_ = dlmread('results_global_ref_0_inc_80/independent_scalars.dat');
r4_ = dlmread('results_global_ref_1_inc_20/independent_scalars.dat');
r5_ = dlmread('results_global_ref_2_inc_20/independent_scalars.dat');

r1=[];
for i=1:length(r1_)-1
  r1(i, :) = 0.5 * (r1_(i, :) + r1_(i+1, :));
end

r2=[];
for i=1:length(r2_)-1
  r2(i, :) = 0.5 * (r2_(i, :) + r2_(i+1, :));
end

r3=[];
for i=1:length(r3_)-1
  r3(i, :) = 0.5 * (r3_(i, :) + r3_(i+1, :));
end

r4=[];
for i=1:length(r4_)-1
  r4(i, :) = 0.5 * (r4_(i, :) + r4_(i+1, :));
end

r5=[];
for i=1:length(r5_)-1
  r5(i, :) = 0.5 * (r5_(i, :) + r5_(i+1, :));
end

hold on; grid on;

plot(r1(:,1), r1(:,2), 'ko-');
plot(r1(:,1), r1(:,3), 'ro-');

plot(r2(:,1), r2(:,2), 'kx-');
plot(r2(:,1), r2(:,3), 'rx-');

plot(r3(:,1), r3(:,2), 'kv-');
plot(r3(:,1), r3(:,3), 'rv-');

%plot(r4(:,1), r4(:,2), 'k^-');
%plot(r4(:,1), r4(:,3), 'r^-');

%plot(r5(:,1), r5(:,2), 'ks-');
%plot(r5(:,1), r5(:,3), 'rs-');