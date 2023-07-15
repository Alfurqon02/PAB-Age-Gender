package com.ml.projects.age_genderdetection

import android.graphics.Bitmap
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class GenderClassificationModel {

    private val inputImageSize = 128

    private val inputImageProcessor =
            ImageProcessor.Builder()
                    .add( ResizeOp( inputImageSize , inputImageSize , ResizeOp.ResizeMethod.BILINEAR ) )
                    .add( NormalizeOp( 0f , 255f ) )
                    .build()

    var inferenceTime : Long = 0

    var interpreter : Interpreter? = null

    suspend fun predictGender( image : Bitmap ) = withContext( Dispatchers.Default ) {
        val start = System.currentTimeMillis()
        val tensorInputImage = TensorImage.fromBitmap( image )
        val genderOutputArray = Array( 1 ){ FloatArray( 2 ) }
        val processedImageBuffer = inputImageProcessor.process( tensorInputImage ).buffer
        interpreter?.run(
            processedImageBuffer,
            genderOutputArray
        )
        inferenceTime = System.currentTimeMillis() - start
        return@withContext genderOutputArray[ 0 ]
    }


}