package org.pytorch.imagesegmentation;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.view.TextureView;
import android.view.ViewStub;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;


public class LiveVideoClassificationActivity extends AbstractCameraXActivity<LiveVideoClassificationActivity.AnalysisResult> {
    private Module mModule = null;
    private int mFrameCount = 0;
    private FloatBuffer inTensorBuffer;
    private TextView mFpsText;
    private Handler mainHandler = new Handler(Looper.getMainLooper());




    static class AnalysisResult {
        private final Bitmap cameraFrame;
        private final Bitmap segmentationMask;

        public AnalysisResult(Bitmap cameraFrame, Bitmap segmentationMask) {
            this.cameraFrame = cameraFrame;
            this.segmentationMask = segmentationMask;
        }

        public Bitmap getCameraFrame() {
            return cameraFrame;
        }

        public Bitmap getSegmentationMask() {
            return segmentationMask;
        }
    }

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_live_video_classification;
    }



    @Override
    protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
        //set texture view to display the result
        CustomCameraView customCameraView = findViewById(R.id.custom_camera_view);
        mFpsText = findViewById(R.id.textView);
        mFpsText.setTextColor(Color.parseColor("#FF0000"));



        // Update the custom view with the new camera frame and segmentation mask
        customCameraView.setCameraFrame(result.getCameraFrame(), true);
        customCameraView.setSegmentationMask(result.getSegmentationMask());
    }

    private Bitmap imgToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        Bitmap result = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
        return result;
    }

    private Bitmap rotateBitmap(Bitmap bitmap, int rotationDegrees) {
        Matrix matrix = new Matrix();
        matrix.postRotate(rotationDegrees);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    private Bitmap centerCropToSquare(Bitmap bitmap) {
        // Dimensions for the resulting square
        int squareSize = Math.min(bitmap.getWidth(), bitmap.getHeight());

        // Calculating the top left corner of the crop area
        int x = 0; // No cropping from sides as the width is already 480
        int y = (bitmap.getHeight() - squareSize) / 2; // Cropping from top and bottom

        // Creating the cropped bitmap
        Bitmap croppedBitmap = Bitmap.createBitmap(bitmap, x, y, squareSize, squareSize);

        return croppedBitmap;
    }



    @Override
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
        if (mModule == null) {
            try {
                mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "segresnet.ptl"));
            } catch (IOException e) {
                return null;
            }
        }

        Bitmap bitmap = imgToBitmap(image.getImage());

        bitmap = rotateBitmap(bitmap, rotationDegrees);

        bitmap = centerCropToSquare(bitmap);

        System.out.println("Nuren "+bitmap.getWidth()+" "+bitmap.getHeight());

        bitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true);


        float[] mean = {0.0F, 0.0F, 0.0F};
        float[] std = {1.0F, 1.0F, 1.0F};
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                mean, std);


        final long startTime = SystemClock.elapsedRealtime();
        long inferenceTime = 0;
        Map<String, IValue> outTensors = mModule.forward(IValue.from(inputTensor)).toDictStringKey();
        inferenceTime = SystemClock.elapsedRealtime() - startTime;
        Log.d("ImageSegmentation", "inference time (ms): " + inferenceTime);

        final float fps = 1000.f / inferenceTime;
        System.out.println("Nuren222 "+fps);
        mainHandler.post(new Runnable() {
            @Override
            public void run() {
                if (mFpsText != null) {
                    mFpsText.setText("FPS: " + String.format("%.2f", fps));
                }
            }
        });

        final Tensor outputTensor = outTensors.get("out").toTensor();

        final float[] scores = outputTensor.getDataAsFloatArray();
        //java.lang.IllegalStateException: Tensor of type Tensor_int64 cannot return data as float array.
//        final long[] scores = outputTensor.getDataAsLongArray();
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        int[] intValues = new int[width * height];
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                double maxnum = 0.95;
                float score = scores[0 * (width * height) + j * width + k];
                if (score > maxnum) {
                    intValues[j * width + k] = 0xFFFF0000;
                } else
                    intValues[j * width + k] = 0xFF000000;
            }
        }


        Bitmap bmpSegmentation = Bitmap.createScaledBitmap(bitmap, width, height, true);
        Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
        outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());
        final Bitmap transferredBitmap = Bitmap.createScaledBitmap(outputBitmap, bitmap.getWidth(), bitmap.getHeight(), true);
//        image.close();
        System.out.println("Nuren111");

        return new AnalysisResult(bitmap, transferredBitmap);

    }
}
