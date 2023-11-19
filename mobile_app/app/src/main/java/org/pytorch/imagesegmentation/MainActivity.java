package org.pytorch.imagesegmentation;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.VideoView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;





public class MainActivity extends AppCompatActivity implements Runnable {

    private CustomCameraView mCustomCameraView;
    private Uri mVideoUri;
    private final String[] mTestVideos = {"video1", "video2", "video3"};
    private int mTestVideoIndex = 0;
    private final String TAG = MainActivity.class.getSimpleName();
    private Thread mThread;
    private Button mButtonSegment;
    private Button mButtonSelect;
    private Button mButtonSegresnet;
    private Button mButtonResunet;
    private Button mButtonMobilenet;
    private Button mButtonStop;
    private Button mButtonLive;
    private TextView mTextView;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Bitmap mBitmapStart = null;
    private Module mModule1 = null;
    private Module mModule2 = null;
    private Module mModule3 = null;
    private String mImagename = "11.jpeg";
    private boolean mStopThread;


    // see http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html for the list of classes with indexes
    private static final int CLASSNUM = 2;

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
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }


    private Uri getMedia(String mediaName) {
        return Uri.parse("android.resource://" + getPackageName() + "/raw/" + mediaName);
    }

    private void stopVideo() {
        mButtonSegment.setEnabled(true);
        mStopThread = true;

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && data != null) {
            Uri selectedMediaUri = data.getData();
            if (selectedMediaUri.toString().contains("mp4")) {
                mButtonSegment.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonSegment.setText(getString(R.string.run_model));
                mVideoUri = selectedMediaUri;
                mStopThread = false;
                Thread thread = new Thread(MainActivity.this);
                thread.start();

            }
        }
    }



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            mBitmapStart = BitmapFactory.decodeStream(getAssets().open(mImagename));
            mBitmapStart = Bitmap.createScaledBitmap(mBitmapStart, 400, 256, true);


        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error reading assets", e);
            finish();
        }

        mTextView = findViewById(R.id.textView2);
        mTextView.setTextColor(Color.parseColor("#FF0000"));
        mTextView.setVisibility(View.INVISIBLE);

        mButtonSegresnet = findViewById(R.id.segresnet);
        mButtonMobilenet = findViewById(R.id.mobilenet);
        mButtonResunet = findViewById(R.id.resunet);
        mButtonSegresnet.setVisibility(View.INVISIBLE);
        mButtonMobilenet.setVisibility(View.INVISIBLE);
        mButtonResunet.setVisibility(View.INVISIBLE);
        mCustomCameraView = findViewById(R.id.custom_camera_view);
        mCustomCameraView.setSegmentationMask(mBitmapStart);
        mCustomCameraView.setCameraFrame(mBitmapStart, false);



        mButtonLive = findViewById(R.id.liveButton);
        mButtonLive.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, LiveVideoClassificationActivity.class);
                startActivity(intent);
                mStopThread = true;
                System.out.println("LIVE BUTTON CLICKED");

            }
        });


        mButtonSelect = findViewById(R.id.restartButton);

        mButtonSelect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {

                Intent pickIntent = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                pickIntent.setType("video/*");
                startActivityForResult(pickIntent, 1);
            }
        });
        mButtonSegment = findViewById(R.id.segmentButton);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
        mButtonStop = findViewById(R.id.stopButton);
        mButtonStop.setVisibility(View.INVISIBLE);

        mButtonStop.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                stopVideo();
            }
        });

        mButtonSegment.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonSegment.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonSegment.setText(getString(R.string.run_model));
                mVideoUri = getMedia(mTestVideos[mTestVideoIndex]);
                mStopThread = false;
                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });
        try {
            mModule1 = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "segresnet.ptl"));
//            mModule2 = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "res_unet.ptl"));
//            mModule3 = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "mobilenet_v3.ptl"));

        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error reading assets", e);
            finish();
        }
        //make videoview invisible


    }

    //function to segment image

    private void segmentImage() {

        float[] mean = {0.0F, 0.0F, 0.0F};
        float[] std = {1.0F, 1.0F, 1.0F};
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap,
                mean, std);

        final float[] inputs = inputTensor.getDataAsFloatArray();


        final long startTime = SystemClock.elapsedRealtime();

        Map<String, IValue> outTensors = mModule1.forward(IValue.from(inputTensor)).toDictStringKey();
        System.out.println("Nuren "+ " forward");
        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
        Log.d("ImageSegmentation", "inference time (ms): " + inferenceTime);

        final Tensor outputTensor = outTensors.get("out").toTensor();

        final float[] scores = outputTensor.getDataAsFloatArray();
        //java.lang.IllegalStateException: Tensor of type Tensor_int64 cannot return data as float array.
//        final long[] scores = outputTensor.getDataAsLongArray();
        int width = mBitmap.getWidth();
        int height = mBitmap.getHeight();

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


        Bitmap bmpSegmentation = Bitmap.createScaledBitmap(mBitmap, width, height, true);
        Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
        outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());
        final Bitmap transferredBitmap = Bitmap.createScaledBitmap(outputBitmap, mBitmap.getWidth(), mBitmap.getHeight(), true);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mTextView.setVisibility(View.VISIBLE);
                mTextView.setText("FPS: " + String.format("%.2f", 1000.f / inferenceTime));
//                mImageView.setImageBitmap(transferredBitmap);
//                mImageView2.setImageBitmap(mBitmap);
                mCustomCameraView.setVisibility(View.VISIBLE);
                mCustomCameraView.setCameraFrame(mBitmap, false);
                mCustomCameraView.setSegmentationMask(transferredBitmap);
                mButtonSelect.setVisibility(View.INVISIBLE);
                mButtonStop.setVisibility(View.VISIBLE);

            }
        });
    }

    @Override
    public void run() {

        //get stream from mirroring device


        MediaMetadataRetriever mmr = new MediaMetadataRetriever();
        mmr.setDataSource(this.getApplicationContext(), mVideoUri);

        //get frame from mVideoView

        //get each frame from video
        int frameCount = Integer.parseInt(mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT));
        //get fps from video

        //int frameRate = Integer.parseInt(mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE));
        //framerate not working. hardcode to 30
        int frameRate = 30;
        int frameTime = 1000 / frameRate;
        int frameIndex = 0;
        while ((frameIndex < frameCount) && !mStopThread)  {
            Bitmap frame = mmr.getFrameAtTime(frameIndex * frameTime * 1000, MediaMetadataRetriever.OPTION_CLOSEST);
            if (frame != null) {
                Bitmap resizedFrame = Bitmap.createScaledBitmap(frame, 256, 256, true);
                mBitmap = resizedFrame;
                segmentImage();
            }
            else if(frame == null && frameIndex != 0)
                break;
            frameIndex++;
        }
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mTextView.setVisibility(View.INVISIBLE);
                mCustomCameraView.setCameraFrame(mBitmapStart, false);
                mCustomCameraView.setSegmentationMask(mBitmapStart);
                mButtonSegment.setVisibility(View.VISIBLE);
                mButtonSegment.setEnabled(true);
                mButtonSelect.setVisibility(View.VISIBLE);
                mButtonStop.setVisibility(View.INVISIBLE);
                mProgressBar.setVisibility(ProgressBar.INVISIBLE);
                mButtonSegment.setText(getString(R.string.segment));

            }
        });


    }

}
