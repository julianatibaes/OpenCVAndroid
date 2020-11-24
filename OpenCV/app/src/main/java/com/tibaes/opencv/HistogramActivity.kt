package com.tibaes.opencv

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.util.Log
import android.view.MenuItem
import android.view.SurfaceView
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.opencv.core.*
import org.opencv.core.Core.LUT
import org.opencv.core.CvType.CV_8UC1
import org.opencv.imgproc.Imgproc
import kotlin.math.roundToInt
import org.opencv.core.MatOfByte
import org.opencv.core.Mat

import androidx.core.app.ComponentActivity.ExtraData

import androidx.core.content.ContextCompat.getSystemService

import android.icu.lang.UCharacter.GraphemeClusterBreak.T
import android.media.Image
import kotlinx.android.synthetic.main.activity_histogram.*
import org.opencv.android.*
import org.opencv.imgcodecs.Imgcodecs
import java.io.ByteArrayInputStream


class HistogramActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private val VIEW_MODE_RGBA = 0
    private val VIEW_MODE_GRAY = 1
    private var mViewMode = VIEW_MODE_RGBA

    private val viewFinder by lazy { findViewById<JavaCameraView>(R.id.cameraViewHist) }
    lateinit var cvBaseLoaderCallback: BaseLoaderCallback

    lateinit var rgbaMat: Mat
    lateinit var grayMat: Mat

    private var isCameraFront = true
    private var isImageColored = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_histogram)

        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.setHomeButtonEnabled(true)

        checkOpenCV(this)

        viewFinder.visibility = SurfaceView.VISIBLE
        viewFinder.setCameraIndex(CameraCharacteristics.LENS_FACING_BACK)
        viewFinder.setCvCameraViewListener(this)

        cvBaseLoaderCallback = object : BaseLoaderCallback(this) {
            override fun onManagerConnected(status: Int) {

                when (status) {
                    SUCCESS -> {
                        MainActivity.lgi(HistogramActivity.OPENCV_SUCCESSFUL)
                        viewFinder.enableView()
                    }

                    else -> super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when(item.itemId){
            android.R.id.home -> {
                startActivity(Intent(this, MainActivity::class.java))
                finishAffinity()}
        }
        return true
    }

    override fun onBackPressed() {
        startActivity(Intent(this, MainActivity::class.java))
        finishAffinity()
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
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        rgbaMat = Mat(width, height, CvType.CV_8UC4)
        grayMat = Mat(width, height, CvType.CV_8UC1)
    }

    override fun onCameraViewStopped() {
        rgbaMat.release()
        grayMat.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        rgbaMat = inputFrame!!.gray()
        grayMat= inputFrame!!.gray()
        val hist_w = 150
        val hist_h = 150
        val histSize = 256
        val bin_w = (hist_w.toDouble() / histSize).roundToInt()
        val histImage = Mat(hist_h, hist_w, CvType.CV_8UC3, Scalar(0.0, 0.0, 0.0))
        Core.normalize(inputFrame.gray(), reduceColors(rgbaMat, 0, 0, 50), 0.0, histImage.rows().toDouble(), Core.NORM_MINMAX, -1,  Mat())
        val hist_b = reduceColors(rgbaMat, 0,0,  50)

        for (i in 1..histSize) {
            Imgproc.line(
                histImage,
                Point(
                    (bin_w * (i - 1)).toDouble(),
                    (hist_h - hist_b.get(i - 1, 0)[0].roundToInt()).toDouble()
                ),
                Point(
                    (bin_w * i).toDouble(),
                    (hist_h - hist_b.get(i, 0)[0].roundToInt()).toDouble()
                ),
                Scalar(255.0, 0.0, 0.0),
                2,
                8,
                0
            )
        }
        histogram.setImageBitmap(mat2Image(histImage))

       /* when (mViewMode) {
            VIEW_MODE_RGBA -> {
                Core.normalize(reduceColors(rgbaMat, 0, 50, 0), reduceColors(rgbaMat, 0, 50, 0),
                    0.0, histImage.rows().toDouble(), Core.NORM_MINMAX, -1, Mat())
                Core.normalize(reduceColors(rgbaMat, 50, 0, 0), reduceColors(rgbaMat, 50, 0, 0),
                    0.0, histImage.rows().toDouble(), Core.NORM_MINMAX, -1, Mat())

                val hist_b = reduceColors(rgbaMat, 0,0,  50)
                val hist_g = reduceColors(rgbaMat, 0,50,  0)
                val hist_r = reduceColors(rgbaMat, 50,0,  0)

                for (i in 1..150){
                    Imgproc.line(
                        histImage,
                        Point(
                            (bin_w * (i - 1)).toDouble(),
                            (hist_h - hist_b.get(i - 1, 0)[0].roundToInt()).toDouble()
                        ),
                        Point(
                            (bin_w * i).toDouble(),
                            (hist_h - hist_b.get(i, 0)[0].roundToInt()).toDouble()
                        ),
                        Scalar(255.0, 0.0, 0.0),
                        2,
                        8,
                        0
                    )
                    Imgproc.line(
                        histImage,
                        Point(
                            (bin_w * (i - 1)).toDouble(),
                            (hist_h - hist_g.get(i - 1, 0)[0].roundToInt()).toDouble()
                        ),
                        Point(
                            (bin_w * i).toDouble(),
                            (hist_h - hist_g.get(i, 0)[0].roundToInt()).toDouble()
                        ),
                        Scalar(0.0, 255.0, 0.0),
                        2,
                        8,
                        0
                    )
                    Imgproc.line(
                        histImage,
                        Point(
                            (bin_w * (i - 1)).toDouble(),
                            (hist_h - hist_r.get(i - 1, 0)[0].roundToInt()).toDouble()
                        ),
                        Point(
                            (bin_w * i).toDouble(),
                            (hist_h - hist_r.get(i, 0)[0].roundToInt()).toDouble()
                        ),
                        Scalar(0.0, 0.0, 255.0),
                        2,
                        8,
                        0
                    )
                }

            }
            VIEW_MODE_GRAY -> {
                Imgproc.cvtColor(inputFrame!!.gray(), rgbaMat, Imgproc.COLOR_BGR2GRAY, 4)
            }
        }*/
        return grayMat
    }

    fun switchImageColor(view: View){
        if (isImageColored){
            mViewMode = VIEW_MODE_RGBA
            isImageColored = false
        } else {
            mViewMode = VIEW_MODE_GRAY
            isImageColored = true
        }
    }

    fun createLUT(numColors: Int): Mat? { // When numColors=1 the LUT will only have 1 color which is black.
        if (numColors < 0 || numColors > 256) {
            println("Invalid Number of Colors. It must be between 0 and 256 inclusive.")
            return null
        }
        val lookupTable: Mat = Mat.zeros(Size(1.0, 256.0), CV_8UC1)
        var startIdx = 0
        var x = 0
        while (x < 256) {
            lookupTable.put(x, 0, x.toDouble())
            for (y in startIdx until x) {
                if (lookupTable[y, 0][0] == 0.0) {
                    lookupTable.put(y, 0, *lookupTable[x, 0])
                }
            }
            startIdx = x
            x += (256.0/numColors).toInt()

        }
        return lookupTable
    }

    fun reduceColors(img: Mat, numRed: Int, numGreen: Int, numBlue: Int): Mat{

        val redLUT = createLUT(numRed)
        val greenLUT  = createLUT(numGreen)
        val blueLUT  = createLUT(numBlue)
        val rgb = ArrayList<Mat>(3)

        Core.split(img, rgb)
        LUT(rgb[0], blueLUT, rgb[0])
        LUT(rgb[1], greenLUT, rgb[1])
        LUT(rgb[2], redLUT, rgb[2])
        Core.merge(rgb, img)

        return img
    }

    private fun mat2Image(input: Mat):Bitmap? {
        val rgb = Mat()
        Imgproc.cvtColor(input, rgb, Imgproc.COLOR_BGR2RGB)
        var bmp: Bitmap?= null
        try {
            bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(rgb, bmp)
        } catch (e: CvException){
            Log.d("Exception", e.message)
        }
        return bmp
    }

    companion object {

        val TAG = "MYLOG " + MainActivity::class.java.simpleName
        fun lge(s: String) = Log.e(TAG, s)
        fun lgi(s: String) = Log.i(TAG, s)

        fun shortMsg(context: Context, s: String) =
            Toast.makeText(context, s, Toast.LENGTH_SHORT).show()

        private const val OPENCV_SUCCESSFUL = "OpenCV Loaded Successfully!"
        private const val OPENCV_FAIL = "Could not load OpenCV!!!"

        private fun checkOpenCV(context: Context) =
            if (OpenCVLoader.initDebug()) {
                shortMsg(context, OPENCV_SUCCESSFUL)
                lgi("OpenCV started...")
            } else {
                shortMsg(context, OPENCV_FAIL)
                lge(OPENCV_FAIL)
            }
    }
}
