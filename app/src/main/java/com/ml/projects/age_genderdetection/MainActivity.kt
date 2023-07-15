package com.ml.projects.age_genderdetection

import android.app.ProgressDialog
import android.content.Intent
import android.graphics.*
import android.graphics.drawable.ColorDrawable
import android.media.ExifInterface
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.content.FileProvider
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.io.IOException
import kotlin.math.floor


class MainActivity : AppCompatActivity() {

    private val realTimeOpts = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .build()
    private val firebaseFaceDetector = FaceDetection.getClient(realTimeOpts)

    private lateinit var sampleImageView : ImageView
    private lateinit var infoTextView : TextView
    private lateinit var ageOutputTextView : TextView
    private lateinit var genderOutputTextView : TextView
    private lateinit var inferenceSpeedTextView : TextView
    private lateinit var resultsLayout : ConstraintLayout
    private lateinit var progressDialog : ProgressDialog

    private val coroutineScope = CoroutineScope( Dispatchers.Main )

    private val REQUEST_IMAGE_CAPTURE = 101
    private val REQUEST_IMAGE_SELECT = 102
    private lateinit var currentPhotoPath : String

    lateinit var ageModelInterpreter: Interpreter
    lateinit var genderModelInterpreter: Interpreter
    private lateinit var ageEstimationModel: AgeEstimationModel
    private lateinit var genderClassificationModel: GenderClassificationModel
    private var useNNApi : Boolean = false
    private var useGpu : Boolean = false
    private val compatList = CompatibilityList()
    private val modelNames = arrayOf(
        "Age/Gender Detection Model ( Quantized ) "
    )
    private val modelFilenames = arrayOf(
        arrayOf("model_age_q.tflite", "model_gender_q.tflite"),
    )
    private var modelFilename = arrayOf( "model_age_q.tflite", "model_gender_q.tflite" )

    private val shift = 5

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        sampleImageView = findViewById(R.id.sample_input_imageview)
        infoTextView = findViewById( R.id.info_textView )
        ageOutputTextView = findViewById( R.id.age_output_textView )
        genderOutputTextView = findViewById( R.id.gender_output_textview )
        resultsLayout = findViewById( R.id.results_layout )
        inferenceSpeedTextView = findViewById( R.id.inference_speed_textView )

        progressDialog = ProgressDialog( this )
        progressDialog.setCancelable( false )
        progressDialog.setMessage( "Searching for faces ...")

        showModelInitDialog()

    }

    fun openCamera( v: View ) {
        dispatchTakePictureIntent()
    }

    fun selectImage( v : View ) {
        dispatchSelectPictureIntent()
    }

    fun reInitModel( v : View ) {
        showModelInitDialog()
    }

    private fun showModelInitDialog() {

        val alertDialogBuilder = AlertDialog.Builder(this)
        alertDialogBuilder.setCancelable( false )
        val dialogView = layoutInflater.inflate(R.layout.model_init_dialog, null)

        val initModelButton : Button = dialogView.findViewById(R.id.init_model_button)
        val closeButton : Button = dialogView.findViewById( R.id.close_button )
        val selectModelSpinner : Spinner = dialogView.findViewById(R.id.select_model_spinner)

        alertDialogBuilder.setView(dialogView)
        val dialog = alertDialogBuilder.create()
        dialog.window?.setBackgroundDrawable( ColorDrawable(Color.TRANSPARENT ) )
        dialog.show()

        val spinnerAdapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, modelNames)
        selectModelSpinner.adapter = spinnerAdapter
        selectModelSpinner.setSelection(0)
        selectModelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View, position: Int, id: Long) {
                modelFilename = modelFilenames[ position ]
            }
            override fun onNothingSelected(parent: AdapterView<*>) {
            }
        }

        closeButton.setOnClickListener {
            dialog.dismiss()
            finish()
        }
        initModelButton.setOnClickListener {
            val options = Interpreter.Options().apply {
                if ( useGpu ) {
                    addDelegate(GpuDelegate( compatList.bestOptionsForThisDevice ) )
                }
                if ( useNNApi ) {
                    addDelegate(NnApiDelegate())
                }
            }
            coroutineScope.launch {
                initModels(options)
            }
            dialog.dismiss()
        }

    }

    private suspend fun initModels(options: Interpreter.Options) = withContext( Dispatchers.Default ) {
        ageModelInterpreter = Interpreter(FileUtil.loadMappedFile( applicationContext , modelFilename[0]), options )
        genderModelInterpreter = Interpreter(FileUtil.loadMappedFile( applicationContext , modelFilename[1]), options )
        withContext( Dispatchers.Main ){
            ageEstimationModel = AgeEstimationModel().apply {
                interpreter = ageModelInterpreter
            }
            genderClassificationModel = GenderClassificationModel().apply {
                interpreter = genderModelInterpreter
            }
            Toast.makeText( applicationContext , "Models initialized." , Toast.LENGTH_LONG ).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        ageModelInterpreter.close()
        genderModelInterpreter.close()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if ( resultCode == RESULT_OK && requestCode == REQUEST_IMAGE_CAPTURE ) {
            var bitmap = BitmapFactory.decodeFile( currentPhotoPath )
            val exifInterface = ExifInterface( currentPhotoPath )
            bitmap =
                when (exifInterface.getAttributeInt( ExifInterface.TAG_ORIENTATION , ExifInterface.ORIENTATION_UNDEFINED )) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap( bitmap , 90f )
                    ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap( bitmap , 180f )
                    ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap( bitmap , 270f )
                    else -> bitmap
                }
            progressDialog.show()
            detectFaces( bitmap!! )
        }
        else if ( resultCode == RESULT_OK && requestCode == REQUEST_IMAGE_SELECT ) {
            val inputStream = contentResolver.openInputStream( data?.data!! )
            val bitmap = BitmapFactory.decodeStream( inputStream )
            inputStream?.close()
            progressDialog.show()
            detectFaces( bitmap!! )
        }
    }

    private fun detectFaces(image: Bitmap) {
        val inputImage = InputImage.fromBitmap(image, 0)
        firebaseFaceDetector.process(inputImage)
                .addOnSuccessListener { faces ->
                    if ( faces.size != 0 ) {
                        sampleImageView.setImageBitmap(cropToBBox(image, faces[0].boundingBox))
                        coroutineScope.launch {

                            // Predict the age and the gender.
                            val age = ageEstimationModel.predictAge(cropToBBox(image, faces[0].boundingBox))
                            val gender = genderClassificationModel.predictGender(cropToBBox(image, faces[0].boundingBox))

                            ageOutputTextView.text = floor( age.toDouble() ).toInt().toString()
                            genderOutputTextView.text = if ( gender[ 0 ] > gender[ 1 ] ) { "Male" } else { "Female" }
                            resultsLayout.visibility = View.VISIBLE
                            infoTextView.visibility = View.GONE
                            progressDialog.dismiss()
                        }
                    }
                    else {
                        progressDialog.dismiss()
                        val dialog = AlertDialog.Builder( this ).apply {
                            title = "No Faces Found"
                            setMessage( "Face not Found at the image. " +
                                    "Try clicking another image or improve the lightning or the device rotation." )
                            setPositiveButton( "OK") { dialog, which ->
                                dialog.dismiss()
                            }
                            setCancelable( false )
                            create()
                        }
                        dialog.show()
                    }


                }
    }


    private fun cropToBBox(image: Bitmap, bbox: Rect) : Bitmap {
        return Bitmap.createBitmap(
            image,
            bbox.left - 0 * shift,
            bbox.top + shift,
            bbox.width() + 0 * shift,
            bbox.height() + 0 * shift
        )
    }


    private fun createImageFile() : File {
        val imagesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("image", ".jpg", imagesDir).apply {
            currentPhotoPath = absolutePath
        }
    }

    private fun dispatchSelectPictureIntent() {
        val selectPictureIntent = Intent( Intent.ACTION_OPEN_DOCUMENT ).apply {
            type = "image/*"
            addCategory( Intent.CATEGORY_OPENABLE )
        }
        startActivityForResult( selectPictureIntent , REQUEST_IMAGE_SELECT )
    }

    private fun dispatchTakePictureIntent() {
        val takePictureIntent = Intent( MediaStore.ACTION_IMAGE_CAPTURE )
        if ( takePictureIntent.resolveActivity( packageManager ) != null ) {
            val photoFile: File? = try {
                createImageFile()
            }
            catch (ex: IOException) {
                null
            }
            photoFile?.also {
                val photoURI = FileProvider.getUriForFile(
                    this,
                    "com.ml.projects.age_genderdetection", it
                )
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            }
        }
    }


    private fun rotateBitmap(original: Bitmap, degrees: Float): Bitmap? {
        val matrix = Matrix()
        matrix.preRotate(degrees)
        return Bitmap.createBitmap(original, 0, 0, original.width, original.height, matrix, true)
    }
}

