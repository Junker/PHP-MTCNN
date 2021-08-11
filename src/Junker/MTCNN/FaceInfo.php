<?php

namespace Junker\MTCNN;

use SplFixedArray;

class FaceInfo
{
	public FaceBox $bbox;
	public SplFixedArray $bbox_reg;
	public SplFixedArray $landmark_reg;
	public SplFixedArray $landmark;

	public function __construct()
	{
		$this->bbox = new FaceBox();
		$this->bbox_reg = new SplFixedArray(4);
		$this->landmark_reg = new SplFixedArray(10);
		$this->landmark = new SplFixedArray(10);
	}
};
