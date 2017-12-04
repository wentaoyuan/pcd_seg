function iou_weighted_ave = part_seg_eva(gt_label_path, pred_label_path)
categories = dir(fullfile(pred_label_path,'0*'));
ncategory = size(categories,1);
iou_all = zeros(ncategory,1);
nmodels = zeros(ncategory,1);
for i=1:ncategory
    pred_models = dir(fullfile(pred_label_path, categories(i).name, '*.seg'));
    gt_models = dir(fullfile(gt_label_path, categories(i).name, '*.seg'));
    assert(length(pred_models)==length(gt_models),'Predictions incomplete!');
    nmodels(i) = size(pred_models,1);
    pred_gt_all = cell(nmodels(i),1);
    for j=1:nmodels(i)
        pred_gt_all{j} = [load(fullfile(pred_label_path, categories(i).name, pred_models(j).name)) ...
            load(fullfile(gt_label_path, categories(i).name, pred_models(j).name))];
    end
    npart = max(cell2mat(pred_gt_all));
    npart = npart(2);
    iou_per_part = zeros(length(pred_gt_all),npart);
    for j=1:npart
        iou_per_part(:,j) = cellfun(@(x) (sum(x(:,1)==j & x(:,2)==j)+eps)/(sum(x(:,1)==j | x(:,2)==j)+eps), pred_gt_all);
    end
    iou_all(i) = mean(iou_per_part(:));
end
iou_weighted_ave = sum(iou_all.*nmodels)/sum(nmodels);
disp(sprintf('Average Per Part IoU on %d Categories: %f',ncategory, iou_weighted_ave))