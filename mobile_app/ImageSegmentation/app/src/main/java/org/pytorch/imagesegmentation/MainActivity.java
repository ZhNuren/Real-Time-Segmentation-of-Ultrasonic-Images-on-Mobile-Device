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
import android.widget.EditText;

import android.widget.ProgressBar;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.util.Arrays;
import java.util.Map;





public class MainActivity extends AppCompatActivity implements Runnable {

    private CustomCameraView mCustomCameraView;
    private Uri mVideoUri;
    private final String[] mTestVideos = {"video1", "video2", "video3"};
    private int mTestVideoIndex = 0;
    private final String TAG = MainActivity.class.getSimpleName();
    private Button mButtonSegment;
    private Button mButtonSelect;
    private Button mButtonConnect;
    private Button mButtonCancel;
    private Button mButtonStop;
    private Button mButtonLive;
    private TextView mTextView;
    private EditText mPlainText;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Bitmap mBitmapStart = null;
    private Module mModule1 = null;
    private String mImagename = "11.jpeg";
    private boolean mStopThread;
    private boolean isReceivingData = false;
    private Socket clientSocket = null;
    private final byte[] FRAME_DELIMITER = "end".getBytes();
    long inferenceTime = 0;
    boolean firstFrame = true;
    float avgFps = 0;
    String server = "";



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



    private void startClient() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    clientSocket = new Socket(server, 8080); //rikhat

                    avgFps = 0;
                    BufferedInputStream inputStream = new BufferedInputStream(clientSocket.getInputStream());

                    isReceivingData = true;
                    ByteArrayOutputStream frameBuffer = new ByteArrayOutputStream();
                    byte[] buffer = new byte[1024]; // Adjust the buffer size as needed

                    int bytesRead;
                    while (isReceivingData && !clientSocket.isClosed()) {
                        bytesRead = inputStream.read(buffer);
                        if (bytesRead == -1) break; // End of stream

                        frameBuffer.write(buffer, 0, bytesRead);

                        byte[] frameData = frameBuffer.toByteArray();
                        int delimiterIndex = findDelimiterIndex(frameData, FRAME_DELIMITER);
                        if (delimiterIndex != -1) { // Check if delimiter was found
                            byte[] frameBytes = Arrays.copyOf(frameData, delimiterIndex); // Extract frame data until delimiter
                            processFrame(frameBytes); // Process the frame

                            // Prepare buffer for next frame (retain data after delimiter)
                            frameBuffer.reset();
                            if (frameData.length > delimiterIndex + FRAME_DELIMITER.length) {
                                frameBuffer.write(frameData, delimiterIndex + FRAME_DELIMITER.length, frameData.length - (delimiterIndex + FRAME_DELIMITER.length));
                            }
                        }
                    }

                    frameBuffer.close();
                    inputStream.close();
                    clientSocket.close();

                } catch (IOException e) {
                    e.printStackTrace();
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
                        mButtonLive.setVisibility(View.VISIBLE);
                        mProgressBar.setVisibility(ProgressBar.INVISIBLE);
                        mButtonSegment.setText(getString(R.string.segment));

                    }
                });
            }
        }).start();
    }


    private int[] computeLPSArray(byte[] pattern) {
        int[] lps = new int[pattern.length];
        int length = 0; // Length of the previous longest prefix suffix

        int i = 1;
        lps[0] = 0; // lps[0] is always 0

        // The loop calculates lps[i] for i = 1 to pattern.length - 1
        while (i < pattern.length) {
            if (pattern[i] == pattern[length]) {
                length++;
                lps[i] = length;
                i++;
            } else { // pattern[i] != pattern[length]
                if (length != 0) {
                    length = lps[length - 1];
                    // Note that we do not increment i here
                } else { // if (length == 0)
                    lps[i] = length;
                    i++;
                }
            }
        }
        return lps;
    }

    private int findDelimiterIndex(byte[] data, byte[] delimiter) {
        if (data.length < delimiter.length) {
            return -1;
        }

        int[] lps = computeLPSArray(delimiter);

        int i = 0; // Index for data[]
        int j = 0; // Index for delimiter[]

        while (i < data.length) {
            if (delimiter[j] == data[i]) {
                j++;
                i++;
            }
            if (j == delimiter.length) {
                return i - j; // Found delimiter at index i - j
            } else if (i < data.length && delimiter[j] != data[i]) {
                if (j != 0) {
                    j = lps[j - 1];
                } else {
                    i = i + 1;
                }
            }
        }
        return -1; // Delimiter not found
    }
    private void processFrame(byte[] frameBytes) {
        Bitmap fbitmap = null;
        Bitmap bitmap = BitmapFactory.decodeByteArray(frameBytes, 0, frameBytes.length);
        if (bitmap!=null) {
             fbitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true);
        }
        final Bitmap finalbitmap = fbitmap;
        if (finalbitmap != null) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    mBitmap = finalbitmap;
                    segmentImage();
                    if (firstFrame) {
                        avgFps = 1000.f/inferenceTime;
                        firstFrame = false;
                    }
                    avgFps = avgFps + 1000.f/inferenceTime;
                    avgFps = avgFps/2;
                }
            });
        } else {
            Log.e(TAG, "Failed to decode bitmap. Data might be incomplete or corrupted.");
        }
    }

    private void stopClient() {
        isReceivingData = false;
        firstFrame = true;
        if (clientSocket != null && !clientSocket.isClosed()) {
            try {
                clientSocket.close();
            } catch (IOException e) {
                Log.e(TAG, "Error closing socket", e);
            }
        }
    }
    private Uri getMedia(String mediaName) {
        return Uri.parse("android.resource://" + getPackageName() + "/raw/" + mediaName);
    }

    private void stopVideo() {
        firstFrame = true;
        mButtonSegment.setEnabled(true);
        mStopThread = true;

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && data != null) {
            Uri selectedMediaUri = data.getData();

            mButtonSegment.setEnabled(false);
            mProgressBar.setVisibility(ProgressBar.VISIBLE);
            mButtonSegment.setText(getString(R.string.run_model));
            mVideoUri = selectedMediaUri;
            mStopThread = false;
            Thread thread = new Thread(MainActivity.this);
            thread.start();


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
        mButtonConnect = findViewById(R.id.connectButton);
        mButtonCancel = findViewById(R.id.cancelButton);
        mPlainText = findViewById(R.id.plainText);
        mTextView = findViewById(R.id.textView2);
        mTextView.setTextColor(Color.parseColor("#FF0000"));
        mTextView.setVisibility(View.INVISIBLE);

        mCustomCameraView = findViewById(R.id.custom_camera_view);
        mCustomCameraView.setSegmentationMask(mBitmapStart);
        mCustomCameraView.setCameraFrame(mBitmapStart, false);



        mButtonLive = findViewById(R.id.liveButton);
        mButtonLive.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonStop.setVisibility(View.INVISIBLE);
                mButtonConnect.setVisibility(View.VISIBLE);
                mButtonLive.setVisibility(View.INVISIBLE);
                mPlainText.setVisibility(View.VISIBLE);
                mButtonCancel.setVisibility(View.VISIBLE);
                mButtonSegment.setVisibility(View.INVISIBLE);
                mButtonSelect.setVisibility(View.INVISIBLE);
            }
        });

        mButtonConnect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonConnect.setVisibility(View.INVISIBLE);
                mButtonSegment.setVisibility(View.VISIBLE);
                mButtonStop.setVisibility(View.VISIBLE);
                mButtonSegment.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonSegment.setText(getString(R.string.run_model));
                server = mPlainText.getText().toString();
                mPlainText.setVisibility(View.INVISIBLE);
                mButtonCancel.setVisibility(View.INVISIBLE);
                startClient();
            }
        });

        mButtonCancel.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonConnect.setVisibility(View.INVISIBLE);
                mButtonLive.setVisibility(View.VISIBLE);
                mPlainText.setVisibility(View.INVISIBLE);
                mButtonCancel.setVisibility(View.INVISIBLE);
                mButtonSegment.setVisibility(View.VISIBLE);
                mButtonSelect.setVisibility(View.VISIBLE);
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
                mButtonSelect.setVisibility(View.VISIBLE);
                mButtonLive.setVisibility(View.VISIBLE);
                mButtonSegment.setVisibility(View.VISIBLE);
                mProgressBar.setVisibility(ProgressBar.INVISIBLE);
                mButtonStop.setVisibility(View.INVISIBLE);
                mButtonSegment.setText(getString(R.string.segment));
                stopVideo();
                stopClient();


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
//            mModule1 = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "mobilenet_v3.ptl"));

        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error reading assets", e);
            finish();
        }

    }

    //function to segment image

    private void segmentImage() {

        final long startTime = SystemClock.elapsedRealtime();

        // Image preprocessing optimized
        float[] mean = {0.0F, 0.0F, 0.0F};
        float[] std = {1.0F, 1.0F, 1.0F};
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap, mean, std);

        // Neural network inference
        Map<String, IValue> outTensors = mModule1.forward(IValue.from(inputTensor)).toDictStringKey();


        final Tensor outputTensor = outTensors.get("out").toTensor();
        final float[] scores = outputTensor.getDataAsFloatArray();

        int width = mBitmap.getWidth();
        int height = mBitmap.getHeight();
        int[] intValues = new int[width * height];

        double maxnum = 0.95;
        // Parallel processing for pixel manipulation (if feasible)
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                float score = scores[0 * (width * height) + j * width + k];
                intValues[j * width + k] = score > maxnum ? 0xFFFF0000 : 0xFF000000;
            }
        }

        Bitmap outputBitmap = mBitmap.copy(mBitmap.getConfig(), true);
        outputBitmap.setPixels(intValues, 0, width, 0, 0, width, height);
        inferenceTime = SystemClock.elapsedRealtime() - startTime;
        Log.d("ImageSegmentation", "inference time (ms): " + inferenceTime);

        // Update UI efficiently
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mTextView.setVisibility(View.VISIBLE);
                mTextView.setText("Average FPS: " + String.format("%.2f", avgFps));
                mCustomCameraView.setVisibility(View.VISIBLE);
                mCustomCameraView.setCameraFrame(mBitmap, true);
                mCustomCameraView.setSegmentationMask(outputBitmap);
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
        avgFps = 0;
        while ((frameIndex < frameCount) && !mStopThread)  {
            Bitmap frame = mmr.getFrameAtTime(frameIndex * frameTime * 1000, MediaMetadataRetriever.OPTION_CLOSEST);
            frame = frame.copy(Bitmap.Config.ARGB_8888, true);
            if (frame != null) {
                Bitmap resizedFrame = Bitmap.createScaledBitmap(frame, 256, 256, true);
                mBitmap = resizedFrame;
                segmentImage();
                if (firstFrame) {
                    avgFps = 1000.f/inferenceTime;
                    firstFrame = false;
                }
                avgFps = avgFps + 1000.f/inferenceTime;
                avgFps = avgFps/2;
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
