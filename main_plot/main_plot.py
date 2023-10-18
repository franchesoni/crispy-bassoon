from dataloaders.pascalvoc import length, load_img_only, get_length_and_load_ind_img_mask_fn, classes
from IISS.large_experiment import get_training_metrics, run_experiment as run_iiss_experiment
from IISS.large_experiment import precompute_for_dataset
from IISS.classify import oracle_SAM
from baselines.sam_iis import get_curves_for_main_plot_oneimageatatime, run_main_plot_experiment, get_curves_for_main_plot
from baselines.samseg_iis import samseg_iis
from dataloaders.pascalvoc import load_img_with_masks_as_batch_fn
from config import device


def main():
    n_seeds = 3
    # this is interactive image segmentation over sam pregenerated masks 
    samseg_iis(load_img_with_masks_as_batch_fn=load_img_with_masks_as_batch_fn, n_images=length, precomputed_dir='runs/pascalvoc', max_clicks_per_image=1, runname=f'runs/samseg_iis_oneimageatatime_N{length}', reset=True)

    # run SAM IIS as baseline, this means clicking on each image-class pair one time. Later we'll compute the curves per class based on these results. This function has to be done only once and is independent of the seeding.
    run_main_plot_experiment(load_img_with_masks_as_batch_fn, n_images=length, dev=False, max_clicks_per_image=1, runname=f'runs/sam_iis_oneimageatatime_N{length}', device=device)

    for pascal_class in classes:
        if pascal_class in ['background', 'void']:
            continue
        # pascal voc
        precompute_for_dataset(load_img_only, length, 'runs/pascalvoc', reset=False)
        subdslen, load_ind_img_mask_fn = get_length_and_load_ind_img_mask_fn(pascal_class, only_pos=True, seed=None)

        N = subdslen  
        print(pascal_class, subdslen)
        precomputed_dir = 'runs/pascalvoc'
        # obtain SAM mask oracle
        oracle_SAM(load_ind_img_mask_fn=load_ind_img_mask_fn, precomputed_dir=precomputed_dir, n_images=N, runname=f'runs/oracle/{pascal_class}', reset=True)

        # run IISS 
        for seed in range(n_seeds):  # the seed modifies tha order in which the images are loaded
            subdslen, seeded_load_ind_img_mask_fn = get_length_and_load_ind_img_mask_fn(pascal_class, only_pos=True, seed=seed)

            runname = f'runs/iiss_all1x_seed{seed}/{pascal_class}'
            # this is not the same as sampresegiis because the clicks are placed taking into account the prediction
            run_iiss_experiment(load_ind_img_mask_fn=seeded_load_ind_img_mask_fn, precomputed_dir=precomputed_dir, n_images=N, max_total_clicks=N, stack_size=1, clicks_per_stack=1, runname=runname, reset=True, single_mode=True)

            # evaluate the performance of IISS over training set
            get_training_metrics(load_ind_img_mask_fn, precomputed_dir, N, runname)

            # evaluate sam iis
            get_curves_for_main_plot_oneimageatatime(runname=f'runs/sam_iis_oneimageatatime_N{length}', ds_length=subdslen, max_total_clicks=N, class_number=classes[pascal_class], seed=seed, load_ind_img_mask_fn=seeded_load_ind_img_mask_fn)

            # the same but for simpleclick (we need to run the experimente from the simpleclick folder first)
            
            get_curves_for_main_plot_oneimageatatime(runname=f'SimpleClick/simpleclick_oneimageatatime_N1464', ds_length=subdslen, max_total_clicks=N, class_number=classes[pascal_class], seed=seed, load_ind_img_mask_fn=seeded_load_ind_img_mask_fn)

            # the same but for samseg iis 
            get_curves_for_main_plot_oneimageatatime(runname=f'runs/samseg_iis_oneimageatatime_N1464', ds_length=subdslen, max_total_clicks=N, class_number=classes[pascal_class], seed=seed, load_ind_img_mask_fn=seeded_load_ind_img_mask_fn)








if __name__ == '__main__':
    main()