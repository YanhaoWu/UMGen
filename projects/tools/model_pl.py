import datetime
import os
import pickle
import numpy as np
import pytorch_lightning as pl
from projects.plugin.misc.misc import BoxOverlap
from projects.tools.decode_map import Imagedecoder, Mapdecoder
from projects.tools.visulize import Visulizer
import torch


class UMGen_PL(pl.LightningModule):
    #  for mutil gpu inference and mutil task inference
    #  not for training
    def __init__(self, config, model, test_loader, device="cpu"):
        super().__init__()
        self.model = model
        self.config = config
        self.test_loader = test_loader

        # define the model inference setting
        num_new_frames = config.num_new_frames
        cond_frames = config.cond_frames
        pred_task = config.pred_task
        infer_from_gt = config.infer_from_gt
        infer_task = config.infer_task
        # Define the path to save the tokens
        self.token_save_path = config.token_save_path
        # Define your model architecture here
        self.inference_setting_dict = dict(
            new_frames=num_new_frames,
            cond_frames=cond_frames,  
            pred_task=pred_task,
            cond_on_par=True,  #
            infer_from_gt=infer_from_gt,  
        )

        # Define the evaluator
        self.evaluator = config.evaluator
        self.box_overlap_anno = BoxOverlap()
        self.box_overlap_predict = BoxOverlap()
        self.ego_overlap = BoxOverlap()
        if "control" in infer_task:
            self.control_test = True
        else:
            self.control_test = False

        self.generate_video_flag = True
        # Define the tokenlizer and normlizer
        self.bbox3d_tokenizer = config.bbox3d_tokenizer
        self.agent_norm = config.agent_norm
        self.ego_pose_tokenizer = config.ego_pose_tokenizer
        self.ego_norm = config.ego_norm

        # Define the init tokens accroad to the infer_task
        self.init_token_mod = config.init_token_mod
        # Define the input frames, it is related to the inference task
        self.input_cond_frames = config.input_cond_frames
        # Define the decoder
        spe_text = config.spe_text + "_" + infer_task
        self.visulizer = Visulizer(
            video_save_path=config.video_save_base_path,
            video_pretext='UMGen',
            width=512,
            height=512,
            project_name='UMGen_infer',
            spe_text=spe_text,
            save_video=config.save_video,
            addtion_ego=True,
            bbox3d_arrow_length_scale=1,
            cond_frames=self.input_cond_frames,
            put_text=config.put_text,
        )
        self.mapdecoder = Mapdecoder(ckpt=config.map_decoder_weights_path, device=device)  #
        self.imagedecoder = Imagedecoder(ckpt=config.image_decoder_weights_path, device=device)

        # other settings
        self.task = {
            "pose_map_bbox3d_image": ["pose", "map", "bbox3d", "image"],
            "pose_map_bbox3d": ["pose", "map", "bbox3d"],
            "pose_map": ["pose", "map"],
            "pose_bbox3d": ["pose", "bbox3d"],
            "bbox3d": ["bbox3d"],
            "bbox3d_trans": ["bbox3d_trans"],
        }

        self.mod_order = self.task[pred_task]

        self.output_path = config.output_path

    def forward(self, x):
        # Define the forward pass of your model here
        pass

    def training_step(self, batch, batch_idx):
        # Define the training step logic here
        pass

    def validation_step(self, batch, batch_idx):
        # Define the validation step logic here
        pass

    def generate_init_tokens(
        self, gt_tokens, input_cond_frames=20, control_tokens=None
    ):
        # we need init tokens for two cases
        # 1 is the FID and mmd 
        init_token_mod = self.init_token_mod
        init_tokens = {}
        if init_token_mod is not None:
            for init_mod in init_token_mod:
                init_tokens[init_mod] = gt_tokens[init_mod][
                    :, input_cond_frames:, ...
                ].clone()  #
                print("Add init token for ", init_mod)

        elif control_tokens is not None:
            for init_mod in control_tokens.keys():
                if len(control_tokens[init_mod].size()) == 3:
                    init_tokens[init_mod] = control_tokens[init_mod].cuda()
                else:
                    init_tokens[init_mod] = control_tokens[init_mod][
                        None, ...
                    ].cuda()
                print("Add control token for ", init_mod)
        else:
            init_tokens = None
            print("No init token is added")

        return init_tokens

    def world_model_evaluate(self, batch, batch_idx):
        # Use UMGen to infer the tokens
        # Batch: Dict
        if self.control_test:
            # control the ego or other object to move
            gt_tokens = batch["dataset_token"]
            control_dict = batch["control_dict"]
            file_name = batch["scene_name"][0]
            try:
                set_spe_text = (
                    self.visulizer.spe_text
                    + str(batch["control_object"].item())
                )
            except:
                set_spe_text = self.visulizer.spe_text + "_ego"
            self.visulizer.spe_text = set_spe_text
            curr_inference_setting = self.inference_setting_dict.copy()
            init_tokens = self.generate_init_tokens(
                gt_tokens.copy(),
                control_tokens=control_dict,
                input_cond_frames=self.input_cond_frames,
            )  # get control tokens for future frames 
            curr_inference_setting["input_cond_tokens"] = batch[
                "dataset_token"
            ]  # get the condition tokens for 20 frames
            if not "no_control" in file_name:
                curr_inference_setting["control_test"] = True
                curr_inference_setting["init_tokens"] = init_tokens
            else:
                curr_inference_setting["control_test"] = False
                curr_inference_setting["init_tokens"] = None

            if "input_cond_frame" in batch.keys():
                curr_inference_setting["input_cond_frames"] = batch[
                    "input_cond_frame"
                ]
            else:
                curr_inference_setting[
                    "input_cond_frames"
                ] = self.input_cond_frames
            # Inference the future tokens with the predifined tokens
            out_tokens = self.model.inference(
                **curr_inference_setting
            )  
            # save the output tokens
            self.save_tokens(out_tokens, file_name=file_name)
            # decode tokens
            (
                bboxes,
                anno_bboxes,
                pose_values,
                real_pose,
                maps,
                decoded_image,
                map_transformed,
            ) = self.decode_tokens(out_tokens.copy(), gt_tokens.copy())
            # visulize the scene
            self.generate_videos(
                bboxes,
                anno_bboxes,
                pose_values,
                real_pose,
                maps,
                decoded_image,
                map_transformed,
                file_name=file_name,
            )

            return None

        else:
            # generate future tokens in freely way
            gt_tokens = batch
            file_name = gt_tokens["file_name"]
            file_name = file_name[0]
            try:
                file_name = file_name.split("/")[-1][:-4]
            except:
                file_name = file_name[0].split("/")[-1][:-4]
            token_save_path = os.path.join(
                self.token_save_path, file_name + "_tokens.pkl"
            )

            if os.path.exists(token_save_path):
                print(file_name, " has been processed")
            else:
                init_tokens = self.generate_init_tokens(
                    gt_tokens, input_cond_frames=self.input_cond_frames
                )  #

                curr_inference_setting = self.inference_setting_dict.copy()
                curr_inference_setting["input_cond_tokens"] = gt_tokens
                curr_inference_setting["init_tokens"] = init_tokens
                curr_inference_setting[
                    "input_cond_frames"
                ] = self.input_cond_frames
                curr_inference_setting["control_test"] = self.control_test

                if curr_inference_setting["new_frames"] == -1:
                    curr_inference_setting["new_frames"] = (
                        gt_tokens["bbox3d"].shape[1]
                        - curr_inference_setting["input_cond_frames"]
                    )  # set to the max length in the pkl

                # inference
                out_tokens = self.model.inference(
                    **curr_inference_setting
                )  
                # save the output tokens
                self.save_tokens(out_tokens, file_name=file_name)
                # decode tokens
                if (
                    self.global_rank != 110
                ):  # 
                    (
                        bboxes,
                        anno_bboxes,
                        pose_values,
                        real_pose,
                        maps,
                        decoded_image,
                        map_transformed,
                    ) = self.decode_tokens(out_tokens.copy(), gt_tokens.copy())
                else:
                    # savefile_name
                    self.save_undecoded_tokens_name(file_name)

 
                if self.generate_video_flag or batch_idx % 100 == 0:
                    # visulize the scene
                    self.generate_videos(
                        bboxes,
                        anno_bboxes,
                        pose_values,
                        real_pose,
                        maps,
                        decoded_image,
                        map_transformed,
                        None,
                        None,
                        file_name,
                    )

                return None

    def test_step(self, batch, batch_idx):
        out = self.world_model_evaluate(batch, batch_idx)

    def on_test_epoch_end(self):
        pass

    def generate_videos(
        self,
        bboxes=None,
        anno_bbox=None,
        pose_values=None,
        real_pose=None,
        maps=None,
        decoded_image=None,
        map_transformed=None,
        collision_id_predict=None,
        collision_id_anno=None,
        file_name=None,
    ):
        #  generate videos
        if bboxes is not None:
            bboxes = np.array(bboxes, dtype=object)
        if anno_bbox is not None:
            anno_bbox = np.array(anno_bbox, dtype=object)
        map_dict = None
        if maps is not None:
            map_dict = {"map": maps}
        if map_transformed is not None:
            map_dict["map_trans"] = map_transformed
        self.visulizer.visulize(
            box=bboxes,
            scene_name=file_name,
            pose=pose_values,
            real_pose=real_pose,
            maps=map_dict,
            decoded_image=decoded_image,
            collision=collision_id_predict,
            anno_collision=collision_id_anno,
        )

    def generate_compare_videos(
        self, decode_images, ori_image=None, scene_name="test"
    ):
        # 
        self.visulizer.vis_pred_video(
            decode_images, scene_name, video_type="pred"
        )
        if ori_image is not None:
            if len(ori_image.shape) == 5:  # 带了batch
                ori_image = ori_image[0]
            self.visulizer.vis_pred_video(
                ori_image, scene_name, renormalize=False, video_type="GT"
            )



    def split_tokens(self, tokens, chunk_size=50):
        # chunk the tokens to avoid OOM
        if tokens.shape[1] <= chunk_size:
            return [tokens]
        else:
            split_tokens = []
            for i in range(0, tokens.shape[1], chunk_size):
                split_tokens.append(tokens[:, i : i + chunk_size, ...])
            return split_tokens

    def save_undecoded_tokens_name(self, file_name):
        txt_path = os.path.join(
            self.token_save_path, "undecoded_token.txt"
        )
        with open(txt_path, "a") as f:
            f.write(file_name + "\n")

    def save_tokens(self, out_tokens, file_name):
        token_save_path = os.path.join(
            self.token_save_path, file_name + "_tokens.pkl"
        )
        with open(token_save_path, "wb") as f:
            pickle.dump(out_tokens, f)

    def decode_tokens(self, pred_tokens, gt_tokens=None):
        # decode the tokens to get the bbox3d, pose, map, image
        bboxes = None
        anno_bboxes = None
        maps = None
        decoded_image = None
        map_transformed = None
        current_device = torch.cuda.current_device()

        if self.model is not None:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()

        with torch.cuda.amp.autocast():
            # decode bbox3d
            if "bbox3d" in self.mod_order:
                bbox3d_tokens = pred_tokens["bbox3d"][0, ...].copy()
                pad_mask = bbox3d_tokens == self.bbox3d_tokenizer.pad_token
                bbox3d_tokens[~pad_mask] = np.clip(
                    bbox3d_tokens[~pad_mask],
                    self.bbox3d_tokenizer.start,
                    self.bbox3d_tokenizer.start
                    + self.bbox3d_tokenizer.vocab_size
                    - 1,
                )
                bboxes, bbox_classes = self.bbox3d_tokenizer.decode(
                    bbox3d_tokens, keep_order=True, no_special=True
                )
                bboxes = self.agent_norm.unnormalize_bbox3d(bboxes)

                if gt_tokens is not None:
                    (
                        anno_bboxes,
                        anno_bbox_classes,
                    ) = self.bbox3d_tokenizer.decode(
                        gt_tokens["bbox3d"].detach().cpu().numpy()[0],
                        no_special=True,
                    )
                    # bboxes = [bbox.reshape(bbox.klhape[0], 8, 2) for bbox in bboxes]
                    anno_bboxes = self.agent_norm.unnormalize_bbox3d(
                        anno_bboxes
                    )
                else:
                    anno_bboxes = None

            # get pose values
            pose_values = self.ego_pose_tokenizer.decode(
                pred_tokens["pose"][0, ...].copy()
            )
            pose_values = self.ego_norm.unnormalize_ego(pose_values)  # changed
            if gt_tokens is not None:
                real_pose = self.ego_pose_tokenizer.decode(
                    gt_tokens["pose"].detach().cpu().numpy()[0]
                )
                real_pose = self.ego_norm.unnormalize_ego(real_pose)  # changed
            else:
                real_pose = None

            if not isinstance(pose_values, np.ndarray):  
                pose_values = np.array(pose_values)

            if "map" in pred_tokens:
                map_chunk = 6  # to avoid OOM
                chunked_token = self.split_tokens(
                    pred_tokens["map"], chunk_size=map_chunk
                )
                maps = []
                for token in chunked_token:
                    maps.append(self.mapdecoder.decode_maps(token).cpu())
                maps = torch.cat(maps, dim=0)
                # maps = self.mapdecoder.decode_maps(pred_tokens['map'])
                if "map_transformed" in pred_tokens:
                    map_transformed = self.mapdecoder.decode_maps(
                        pred_tokens["map_transformed"]
                    )

            if "image" in pred_tokens:
                image_chunk = 6  # 
                chunked_token = self.split_tokens(
                    pred_tokens["image"], chunk_size=image_chunk
                )
                images = []
                for token in chunked_token:
                    images.append(self.imagedecoder.decode_images(token).cpu())
                    torch.cuda.empty_cache()  # 
                decoded_image = torch.cat(images, dim=0)
                # decoded_image = self.imagedecoder.decode_images(pred_tokens['image'])

        if self.model is not None:
            torch.cuda.empty_cache()
            self.model = self.model.to(current_device)  # 

        return (
            bboxes,
            anno_bboxes,
            pose_values,
            real_pose,
            maps,
            decoded_image,
            map_transformed,
        )

    def test_dataloader(self):

        return self.test_loader

    def configure_optimizers(self):
        # Define your optimizer and learning rate scheduler here
        pass


if __name__ == "__main__":
    model = UMGen_PL()
    # Instantiate the trainer
    trainer = pl.Trainer()
    # Train the model
    trainer.fit(model)
    pass

