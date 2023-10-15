
from dataloaders.pascalvoc import length, load_img_only, get_length_and_load_ind_img_mask_fn, classes
from IISS.large_experiment import get_training_metrics, run_experiment as run_iiss_experiment
from IISS.large_experiment import precompute_for_dataset
from IISS.classify import oracle_SAM

def main():
    for pascal_class in classes:
        if pascal_class in ['background', 'void']:
            continue
        # pascal voc
        precompute_for_dataset(load_img_only, length, 'runs/pascalvoc', reset=False)
        
        subdslen, load_ind_img_mask_fn = get_length_and_load_ind_img_mask_fn(pascal_class, only_pos=True)

        print(pascal_class, subdslen)
        stack_size, clicks_per_stack = 16, 4
        N = (subdslen // stack_size ) * stack_size  # make it a multiple of stack_size, oups
        precomputed_dir = 'runs/pascalvoc'
        # obtain SAM mask oracle
        oracle_SAM(load_ind_img_mask_fn=load_ind_img_mask_fn, precomputed_dir=precomputed_dir, n_images=N, runname=f'{pascal_class}_oracle_N{N}', reset=True)

        # run IISS 
        runname = f'{pascal_class}_iiss_N{N}_tc{N}_s{stack_size}_cs{clicks_per_stack}'
        run_iiss_experiment(load_ind_img_mask_fn=load_ind_img_mask_fn, precomputed_dir=precomputed_dir, n_images=N, max_total_clicks=N, stack_size=stack_size, clicks_per_stack=clicks_per_stack, runname=runname, reset=True)

        # evaluate the performance of IISS over training set
        get_training_metrics(load_ind_img_mask_fn, precomputed_dir, N, runname)








if __name__ == '__main__':
    main()