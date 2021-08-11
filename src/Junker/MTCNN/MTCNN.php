<?php

namespace Junker\MTCNN;

use SplFixedArray;
use CV\Size;
use CV\Scalar;
use CV\Mat;
use CV\Rect;
use function CV\{resize};
use function CV\DNN\{blobFromImage, blobFromImages, readNetFromCaffe};

class MTCNN
{
	private $PNet;
	private $RNet;
	private $ONet;

	const PNET_STRIDE = 2;
	const PNET_CELL_SIZE = 12;
	const PNET_MAX_DETECT_NUM = 5000;

	const MEAN_VAL = 127.5;
	const STD_VAL = 0.0078125;

	const STEP_SIZE = 128;

	const NMS_METHOD_MIN = 0;
	const NMS_METHOD_UNION = 1;


	private array $candidate_boxes = [];
	private array $total_boxes = [];

	public function __construct()
	{
		$model_dir = __DIR__ . '/../../../model';

		$this->PNet = readNetFromCaffe($model_dir . "/det1_.prototxt", $model_dir . "/det1_.caffemodel");
		$this->RNet = readNetFromCaffe($model_dir . "/det2.prototxt", $model_dir . "/det2-half.caffemodel");
		$this->ONet = readNetFromCaffe($model_dir . "/det3-half.prototxt", $model_dir . "/det3-half.caffemodel");
	}

	private static function compareBBox(FaceInfo $a, FaceInfo $b): bool
	{
		return $a->bbox->score < $b->bbox->score;
	}

	private function BBoxRegression(array &$bboxes)
	{
		foreach ($bboxes as &$bbox_obj)
		{
			$bbox = &$bbox_obj->bbox;
			$bbox_reg = $bbox_obj->bbox_reg;

			$w = $bbox->xmax - $bbox->xmin + 1;
			$h = $bbox->ymax - $bbox->ymin + 1;

			$bbox->xmin += $bbox_reg[0] * $w;
			$bbox->ymin += $bbox_reg[1] * $h;
			$bbox->xmax += $bbox_reg[2] * $w;
			$bbox->ymax += $bbox_reg[3] * $h;
		}
	}

	private function BBoxPad(array &$bboxes, int $width, int $height)
	{
		foreach ($bboxes as &$bbox_obj)
		{
			$bbox = &$bbox_obj->bbox;
			$bbox->xmin = round(max($bbox->xmin, 0));
			$bbox->ymin = round(max($bbox->ymin, 0));
			$bbox->xmax = round(min($bbox->xmax, $width - 1));
			$bbox->ymax = round(min($bbox->ymax, $height - 1));
		}
	}

	private function BBoxPadSquare(array &$bboxes, int $width, int $height)
	{
		foreach ($bboxes as &$bbox_obj)
		{
			$bbox = &$bbox_obj->bbox;
			$w = $bbox->xmax - $bbox->xmin + 1;
			$h = $bbox->ymax - $bbox->ymin + 1;
			$side = $h > $w ? $h : $w;

			$bbox->xmin = round(max($bbox->xmin + ($w - $side) * 0.5, 0));
			$bbox->ymin = round(max($bbox->ymin + ($h - $side) * 0.5, 0));
			$bbox->xmax = round(min($bbox->xmin + $side - 1, $width - 1));
			$bbox->ymax = round(min($bbox->ymin + $side - 1, $height - 1));
		}
	}

	private function generateBBox(Mat $confidence, Mat $reg_box, float $scale, float $thresh)
	{
		$feature_map_w = $confidence->shape[3];
		$feature_map_h = $confidence->shape[2];
		$spatial_size = $feature_map_w * $feature_map_h;

		$this->candidate_boxes = [];

		for ($i = 0; $i<$spatial_size; $i++)
		{
			if ($confidence->dataAt($i + $spatial_size) <= 1-$thresh)
			{
				$y = (int)($i / $feature_map_w);
				$x = ($i - $feature_map_w * $y);

				$faceInfo = new FaceInfo();
				$faceInfo->bbox->xmin = (float)($x * self::PNET_STRIDE) / $scale;
				$faceInfo->bbox->ymin = (float)($y * self::PNET_STRIDE) / $scale;
				$faceInfo->bbox->xmax = (float)($x * self::PNET_STRIDE + self::PNET_CELL_SIZE - 1) / $scale;
				$faceInfo->bbox->ymax = (float)($y * self::PNET_STRIDE + self::PNET_CELL_SIZE - 1) / $scale;

				$faceInfo->bbox_reg[0] = $reg_box->dataAt($i);
				$faceInfo->bbox_reg[1] = $reg_box->dataAt($i + $spatial_size);
				$faceInfo->bbox_reg[2] = $reg_box->dataAt($i + 2 * $spatial_size);
				$faceInfo->bbox_reg[3] = $reg_box->dataAt($i + 3 * $spatial_size);
				$faceInfo->bbox->score = $confidence->dataAt($i + $spatial_size);

				$this->candidate_boxes[] = $faceInfo;
			}
		}

	}


	private function NMS(array $bboxes, float $thresh, int $method)
	{
		$bboxes_nms = [];

		if (count($bboxes) == 0)
			return $bboxes_nms;

		usort($bboxes, [__CLASS__, 'compareBBox']);

		$select_idx = 0;
		$num_bbox = count($bboxes);

		$mask_merged = new SplFixedArray($num_bbox);
		$all_merged = false;

		while (!$all_merged)
		{
			while ($select_idx < $num_bbox && $mask_merged[$select_idx] == 1)
				$select_idx++;

			if ($select_idx == $num_bbox)
			{
				$all_merged = true;
				continue;
			}


			$bboxes_nms[] = $bboxes[$select_idx];
			$mask_merged[$select_idx] = 1;

			$select_bbox = $bboxes[$select_idx]->bbox;
			$area1 = ($select_bbox->xmax - $select_bbox->xmin + 1) * ($select_bbox->ymax - $select_bbox->ymin + 1);
			$x1 = $select_bbox->xmin;
			$y1 = $select_bbox->ymin;
			$x2 = $select_bbox->xmax;
			$y2 = $select_bbox->ymax;

			$select_idx++;

			for ($i = $select_idx; $i < $num_bbox; $i++)
			{
				if ($mask_merged[$i] == 1)
					continue;

				$bbox_i = &$bboxes[$i]->bbox;
				$x = max($x1, $bbox_i->xmin);
				$y = max($y1, $bbox_i->ymin);
				$w = min($x2, $bbox_i->xmax) - $x + 1;
				$h = min($y2, $bbox_i->ymax) - $y + 1;

				if ($w <= 0 || $h <= 0)
					continue;

				$area2 = ($bbox_i->xmax - $bbox_i->xmin + 1) * ($bbox_i->ymax - $bbox_i->ymin + 1);
				$area_intersect = $w * $h;

				switch ($method)
				{
					case self::NMS_METHOD_UNION:
						if ($area_intersect / ($area1 + $area2 - $area_intersect) > $thresh)
							$mask_merged[$i] = 1;
						break;
					case self::NMS_METHOD_MIN:
						if ($area_intersect / min($area1, $area2) > $thresh)
							$mask_merged[$i] = 1;
						break;
					default:
						break;
				}
			}
		}

		return $bboxes_nms;
	}

	private function nextStage(Mat $image, array $pre_stage_res, int $input_w, int $input_h, int $stage_num, float $threshold): array
	{
		$res = [];
		$batch_size = count($pre_stage_res);

		if ($batch_size == 0)
			return $res;

		$input_layer = null;
		$confidence = null;
		$reg_box = null;
		$reg_landmark = null;

		$targets_blobs = [];

		switch ($stage_num)
		{
			case 2:
				break;
			case 3:
				break;
			default:
				return res;
				break;
		}

		$spatial_size = $input_h * $input_w;

		$inputs = [];

		for ($n = 0; $n < $batch_size; ++$n)
		{
			$box = $pre_stage_res[$n]->bbox;
			$rect = new Rect((int)$box->xmin, (int)$box->ymin, (int)($box->xmax - $box->xmin), (int)($box->ymax - $box->ymin));
			$roi = $image->getImageROI($rect);
			resize($roi, $roi, new Size($input_w, $input_h));

			$inputs[] = $roi;
		}

		$blob_input = blobFromImages($inputs, self::STD_VAL, new Size(), new Scalar(self::MEAN_VAL, self::MEAN_VAL, self::MEAN_VAL), false);

		switch ($stage_num)
		{
			case 2:
				$this->RNet->setInput($blob_input, "data");
				$targets_node = ["conv5-2","prob1"];
				$targets_blobs = $this->RNet->forwardMulti($targets_node);
				$confidence = $targets_blobs[1];
				$reg_box = $targets_blobs[0];
				break;
			case 3:
				$this->ONet->setInput($blob_input, "data");
				$targets_node = ["conv6-2","conv6-3","prob1"];
				$targets_blobs = $this->ONet->forwardMulti($targets_node);
				$reg_box = $targets_blobs[0];
				$reg_landmark = $targets_blobs[1];
				$confidence = $targets_blobs[2];
				break;
		}

		for ($k = 0; $k < $batch_size; ++$k)
		{
			if ($confidence->dataAt(2 * $k + 1) >= $threshold)
			{
				$info = new FaceInfo();
				$info->bbox->score = $confidence->dataAt(2 * $k + 1);
				$info->bbox->xmin = $pre_stage_res[$k]->bbox->xmin;
				$info->bbox->ymin = $pre_stage_res[$k]->bbox->ymin;
				$info->bbox->xmax = $pre_stage_res[$k]->bbox->xmax;
				$info->bbox->ymax = $pre_stage_res[$k]->bbox->ymax;

				for ($i = 0; $i < 4; ++$i)
					$info->bbox_reg[$i] = $reg_box->dataAt(4 * $k + $i);

				if ($reg_landmark)
				{
					$w = $info->bbox->xmax - $info->bbox->xmin + 1;
					$h = $info->bbox->ymax - $info->bbox->ymin + 1;

					for ($i = 0; $i < 5; ++$i)
					{
						$info->landmark[2 * $i] = $reg_landmark->dataAt(10 * $k + 2 * $i) * $w + $info->bbox->xmin;
						$info->landmark[2 * $i + 1] = $reg_landmark->dataAt(10 * $k + 2 * $i + 1) * $h + $info->bbox->ymin;
					}
				}

				$res[] = $info;
			}
		}

		return $res;
	}

	private function proposalNet(Mat $img, int $minSize, float $threshold, float $factor): array
	{
		$resized = null;
		$width = $img->cols;
		$height = $img->rows;
		$scale = 12 / $minSize;
		$minWH = min($height, $width) * $scale;
		$scales = [];

		while ($minWH >= 12)
		{
			$scales[] = $scale;
			$minWH *= $factor;
			$scale *= $factor;
		}

		$this->total_boxes = [];

		for ($i = 0; $i < count($scales); $i++)
		{
			$ws = (int)ceil($width * $scales[$i]);
			$hs = (int)ceil($height * $scales[$i]);

			resize($img, $resized, new Size($ws, $hs));

			$inputBlob = blobFromImage($resized, 1/255.0, new Size(), new Scalar(0,0,0), false);

			$this->PNet->setInput($inputBlob, "data");

			$targets_node = ["conv4-2","prob1"];
			$targets_blobs = $this->PNet->forwardMulti($targets_node);
			$prob = $targets_blobs[1];
			$reg = $targets_blobs[0];

			$this->generateBBox($prob, $reg, $scales[$i], $threshold);

			$bboxes_nms = $this->NMS($this->candidate_boxes, 0.5, self::NMS_METHOD_UNION);

			if (count($bboxes_nms) > 0)
				$this->total_boxes = array_merge($bboxes_nms, $this->total_boxes);
		}

		$num_box = count($this->total_boxes);
		$res_boxes = [];

		if ($num_box != 0)
		{
			$res_boxes = $this->NMS($this->total_boxes, 0.7, self::NMS_METHOD_UNION);
			$this->BBoxRegression($res_boxes);
			$this->BBoxPadSquare($res_boxes, $width, $height);
		}

		return $res_boxes;
	}

	public function detect(Mat &$image, int $minSize, array $threshold, float $factor, int $stage = 3)
	{
		$pnet_res = [];
		$rnet_res = [];
		$onet_res = [];

		if ($stage >= 1)
			$pnet_res = $this->proposalNet($image, $minSize, $threshold[0], $factor);

		if ($stage >= 2 && count($pnet_res) > 0)
		{
			if (self::PNET_MAX_DETECT_NUM < count($pnet_res))
				$pnet_res = array_slice($pnet_res, 0, self::PNET_MAX_DETECT_NUM);

			$num = count($pnet_res);
			$size = (int)ceil(1*$num / self::STEP_SIZE);

			for ($iter = 0; $iter < $size; ++$iter)
			{
				$input = array_slice($pnet_res, $iter * self::STEP_SIZE, self::STEP_SIZE);

				$res = $this->nextStage($image, $input, 24, 24, 2, $threshold[1]);
				$rnet_res = array_merge($rnet_res, $res);
			}

			$rnet_res = $this->NMS($rnet_res, 0.4, self::NMS_METHOD_MIN);
			$this->BBoxRegression($rnet_res);
			$this->BBoxPadSquare($rnet_res, $image->cols, $image->rows);

		}

		if ($stage >= 3 && count($rnet_res) > 0)
		{
			$num = count($rnet_res);
			$size = (int)ceil(1 * $num / self::STEP_SIZE);

			for ($iter = 0; $iter < $size; ++$iter)
			{
				$input = array_slice($rnet_res, $iter * self::STEP_SIZE, self::STEP_SIZE);
				$res = $this->nextStage($image, $input, 48, 48, 3, $threshold[2]);

				$onet_res = array_merge($onet_res, $res);
			}

			$this->BBoxRegression($onet_res);
			$onet_res = $this->NMS($onet_res, 0.4, self::NMS_METHOD_MIN);
			$this->BBoxPad($onet_res, $image->cols, $image->rows);

		}

		if ($stage == 1)
			return $pnet_res;
		else if ($stage == 2)
			return $rnet_res;
		else if ($stage == 3)
			return $onet_res;
		else
			return $onet_res;
	}
}
