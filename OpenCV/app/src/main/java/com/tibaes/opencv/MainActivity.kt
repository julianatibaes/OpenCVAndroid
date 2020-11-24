package com.tibaes.opencv

import android.Manifest.permission.*
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.OrientationEventListener
import android.view.SurfaceView
import android.view.WindowManager.LayoutParams.*
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.android.synthetic.main.activity_main.*
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.JavaCameraView
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.resize
import org.opencv.objdetect.CascadeClassifier
import yuku.ambilwarna.AmbilWarnaDialog
import java.io.File
import java.io.FileOutputStream
import java.io.IOException


// Permission vars:
private const val REQUEST_CODE_PERMISSIONS = 111
private val REQUIRED_PERMISSIONS = arrayOf(
    CAMERA,
    WRITE_EXTERNAL_STORAGE,
    READ_EXTERNAL_STORAGE,
    RECORD_AUDIO,
    ACCESS_FINE_LOCATION
)

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private val VIEW_MODE_RGBA = 0
    private val VIEW_MODE_GRAY = 1
    private val VIEW_MODE_CANNY = 2
    private val VIEW_MODE_BW = 3
    private val VIEW_MODE_FEATURES = 5
    private val VIEW_CHOOSE_COLOR = 6
    private val VIEW_GAUSSIANA = 7

    private var mViewMode = VIEW_MODE_RGBA

    // view
    private val viewFinder by lazy { findViewById<JavaCameraView>(R.id.cameraView) }

    lateinit var cvBaseLoaderCallback: BaseLoaderCallback

    // image storage
    lateinit var rgbaMat: Mat
    lateinit var grayMat: Mat
    lateinit var faceMat: Mat
    lateinit var intermediateMat: Mat

    // face library
    var faceDetector: CascadeClassifier? = null
    lateinit var faceDir: File
    var imageRatio = 0.0 // scale down ratio

    var chosenColor = Color.RED
    private var isCameraFront = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.clearFlags(FLAG_FORCE_NOT_FULLSCREEN)
        window.setFlags(FLAG_FULLSCREEN, FLAG_FULLSCREEN)
        window.addFlags(FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        // Request camera permissions
        if (allPermissionsGranted()) {
            checkOpenCV(this)
        } else {
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            )
        }

        checkOpenCV(this)
        viewFinder.visibility = SurfaceView.VISIBLE
        viewFinder.setCameraIndex(CameraCharacteristics.LENS_FACING_BACK)
        viewFinder.setCvCameraViewListener(this)

        cvBaseLoaderCallback = object : BaseLoaderCallback(this) {
            override fun onManagerConnected(status: Int) {

                when (status) {
                    SUCCESS -> {
                        lgi(OPENCV_SUCCESSFUL)

                        loadFaceLib()

                        if (faceDetector!!.empty()) {
                            faceDetector = null
                        } else {
                            faceDir.delete()
                        }
                        viewFinder.enableView()
                    }

                    else -> super.onManagerConnected(status)
                }
            }
        }

        val mOrientationEventListener = object : OrientationEventListener(this) {
            override fun onOrientationChanged(orientation: Int) {
                // Monitors orientation values to determine the target rotation value
                when (orientation) {
                    in 45..134 -> {
                        rotation_tv.text = getString(R.string.n_270_degree)
                    }
                    in 135..224 -> {
                        rotation_tv.text = getString(R.string.n_180_degree)
                    }
                    in 225..314 -> {
                        rotation_tv.text = getString(R.string.n_90_degree)
                    }
                    else -> {
                        rotation_tv.text = getString(R.string.n_0_degree)
                    }
                }

            }
        }
        if (mOrientationEventListener.canDetectOrientation()) {
            mOrientationEventListener.enable();
        } else {
            mOrientationEventListener.disable();
        }

    }

    /**
     * Process result from permission request dialog box, has the request
     * been granted? If yes, start Camera. Otherwise display a toast
     */
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                checkOpenCV(this)
            } else {
                shortMsg(this, PERMISSION_NOT_GRANTED)
                finish()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, cvBaseLoaderCallback)
    }

    override fun onPause() {
        super.onPause()
        viewFinder?.let { viewFinder.disableView() }
    }

    override fun onDestroy() {
        super.onDestroy()
        viewFinder?.let { viewFinder.disableView() }
        if (faceDir.exists()) faceDir.delete()
    }


    /**
     * Check if all permission specified in the manifest have been granted
     */
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    companion object {

        val TAG = "MYLOG " + MainActivity::class.java.simpleName
        fun lgd(s: String) = Log.d(TAG, s)
        fun lge(s: String) = Log.e(TAG, s)
        fun lgi(s: String) = Log.i(TAG, s)

        fun shortMsg(context: Context, s: String) =
            Toast.makeText(context, s, Toast.LENGTH_SHORT).show()

        // messages:
        private const val OPENCV_SUCCESSFUL = "OpenCV Loaded Successfully!"
        private const val OPENCV_FAIL = "Could not load OpenCV!!!"
        private const val OPENCV_PROBLEM = "There's a problem in OpenCV."
        private const val PERMISSION_NOT_GRANTED = "Permissions not granted by the user."

        private fun checkOpenCV(context: Context) =
            if (OpenCVLoader.initDebug()) {
                shortMsg(context, OPENCV_SUCCESSFUL)
                lgi("OpenCV started...")
            } else {
                shortMsg(context, OPENCV_FAIL)
                lge(OPENCV_FAIL)
            }

        // Face model
        private const val FACE_DIR = "facelib"
        private const val FACE_MODEL = "haarcascade_frontalface_alt2.xml"
        private const val byteSize = 4096 // buffer size
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        rgbaMat = Mat(width, height, CvType.CV_8UC4)
        grayMat = Mat(width, height, CvType.CV_8UC1)
        intermediateMat = Mat(width, height, CvType.CV_8UC4)
        faceMat = Mat(width, height, CvType.CV_8UC4)
    }

    override fun onCameraViewStopped() {
        rgbaMat.release()
        grayMat.release()
        intermediateMat.release()
        faceMat.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        when (mViewMode) {
            VIEW_MODE_RGBA -> {
                rgbaMat = inputFrame!!.rgba()
            }
            VIEW_MODE_GRAY -> {
                Imgproc.cvtColor(inputFrame!!.gray(), rgbaMat, Imgproc.COLOR_GRAY2RGBA, 4)
            }
            VIEW_MODE_CANNY -> {
                rgbaMat = inputFrame!!.rgba()
                Imgproc.Canny(inputFrame.gray(), intermediateMat, 80.0, 100.0)
                Imgproc.cvtColor(intermediateMat, rgbaMat, Imgproc.COLOR_GRAY2RGBA, 4)
            }
            VIEW_MODE_FEATURES -> {
                rgbaMat = inputFrame!!.rgba()
                grayMat = inputFrame.gray()
            }
            VIEW_MODE_BW -> {
                rgbaMat = inputFrame!!.rgba()
                Imgproc.cvtColor(rgbaMat, rgbaMat, Imgproc.COLOR_BGR2GRAY)
                Imgproc.threshold(rgbaMat, rgbaMat, 120.0, 255.0, Imgproc.THRESH_BINARY)
            }
            VIEW_CHOOSE_COLOR -> {
               // chooseColor()
            }
            VIEW_GAUSSIANA -> {
                rgbaMat = inputFrame!!.rgba()
                Imgproc.cvtColor(rgbaMat, rgbaMat, Imgproc.COLOR_BGR2GRAY)
                Imgproc.GaussianBlur(rgbaMat, rgbaMat, Size(5.0,5.0), 10.0 )
            }
        }

        faceMat = inputFrame!!.gray()
        //grayMat = get480Image(inputFrame!!.gray())
        imageRatio = 1.0

        // detect face rectangle
        //drawFaceRectangle()

        return rgbaMat
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        return super.onCreateOptionsMenu(menu)
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            R.id.menu_switch_camera ->{
                isCameraFront = if(isCameraFront) {
                    viewFinder.disableView()
                    viewFinder.setCameraIndex(CameraCharacteristics.LENS_FACING_FRONT)
                    viewFinder.setCvCameraViewListener(this)
                    viewFinder.enableView()
                    false
                } else {
                    viewFinder.disableView()
                    viewFinder.setCameraIndex(CameraCharacteristics.LENS_FACING_BACK)
                    viewFinder.setCvCameraViewListener(this)
                    viewFinder.enableView()
                    true
                }
            }
            R.id.menu_color -> {
                mViewMode = VIEW_MODE_RGBA
            }
            R.id.menu_gray -> {
                mViewMode = VIEW_MODE_GRAY
            }
            R.id.menu_canny -> {
                mViewMode = VIEW_MODE_CANNY
            }
            R.id.menu_bw -> {
                mViewMode = VIEW_MODE_BW
            }
            R.id.menu_choose_color -> {
                mViewMode = VIEW_CHOOSE_COLOR
            }
            R.id.menu_gaussiana -> {
                mViewMode = VIEW_GAUSSIANA
            }
            R.id.menu_histogram -> {
                startActivity(Intent(this, HistogramActivity::class.java))
            }
        }
        return true
    }

    private fun loadFaceLib() {
        try {
            val modelInputStream =
                resources.openRawResource(
                    R.raw.haarcascade_frontalface_alt2
                )

            // create a temp directory
            faceDir = getDir(FACE_DIR, Context.MODE_PRIVATE)

            // create a model file
            val faceModel = File(faceDir, FACE_MODEL)

            if (!faceModel.exists()) { // copy model
                // copy model to new face library
                val modelOutputStream = FileOutputStream(faceModel)

                val buffer = ByteArray(byteSize)
                var byteRead = modelInputStream.read(buffer)
                while (byteRead != -1) {
                    modelOutputStream.write(buffer, 0, byteRead)
                    byteRead = modelInputStream.read(buffer)
                }

                modelInputStream.close()
                modelOutputStream.close()
            }

            faceDetector = CascadeClassifier(faceModel.absolutePath)
        } catch (e: IOException) {
            lge("Error loading cascade face model...$e")
        }
    }

    fun drawFaceRectangle() {
        val faceRects = MatOfRect()
        faceDetector!!.detectMultiScale(
            faceMat,
            faceRects
        )

        for (rect in faceRects.toArray()) {
            var x = 0.0
            var y = 0.0
            var w = 0.0
            var h = 0.0

            if (imageRatio.equals(1.0)) {
                x = rect.x.toDouble()
                y = rect.y.toDouble()
                w = x + rect.width
                h = y + rect.height
            } else {
                x = rect.x.toDouble() / imageRatio
                y = rect.y.toDouble() / imageRatio
                w = x + (rect.width / imageRatio)
                h = y + (rect.height / imageRatio)
            }

            Imgproc.rectangle(
                rgbaMat,
                Point(x, y),
                Point(w, h),
                Scalar(255.0, 0.0, 0.0)
            )
        }
    }

    fun ratioTo480(src: Size): Double {
        val w = src.width
        val h = src.height
        val heightMax = 480
        var ratio: Double = 0.0

        if (w > h) {
            if (w < heightMax) return 1.0
            ratio = heightMax / w
        } else {
            if (h < heightMax) return 1.0
            ratio = heightMax / h
        }

        return ratio
    }

    fun get480Image(src: Mat): Mat {
        val imageSize = Size(src.width().toDouble(), src.height().toDouble())
        imageRatio = ratioTo480(imageSize)

        if (imageRatio.equals(1.0)) return src

        val dstSize = Size(imageSize.width * imageRatio, imageSize.height * imageRatio)
        val dst = Mat()
        resize(src, dst, dstSize)
        return dst
    }

    private fun chooseColor() {
        val colorPicker = AmbilWarnaDialog(this@MainActivity, chosenColor,  object: AmbilWarnaDialog.OnAmbilWarnaListener {
            override fun onCancel(dialog: AmbilWarnaDialog) {
            }
            override fun onOk(dialog: AmbilWarnaDialog ,color: Int) {
                chosenColor = color
            }
        })
        colorPicker.show()
    }

}
