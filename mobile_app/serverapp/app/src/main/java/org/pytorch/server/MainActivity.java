package org.pytorch.server;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.view.View;
import android.widget.Button;

import org.pytorch.imagesegmentation.R;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;


public class MainActivity extends AppCompatActivity {

    private MediaProjectionManager projectionManager;
    private static final int REQUEST_CODE_CAPTURE_PERM = 1234;



    @Override
    protected void onCreate(Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        projectionManager = (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);

        Button startServerButton = findViewById(R.id.startserver);
        Button stopServerButton = findViewById(R.id.stopButton);
        stopServerButton.setEnabled(false);

        startServerButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                startServerButton.setEnabled(false);
                stopServerButton.setEnabled(true);
                Intent captureIntent = projectionManager.createScreenCaptureIntent();
                startActivityForResult(captureIntent, REQUEST_CODE_CAPTURE_PERM);

            }
        });
        stopServerButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Stop the MediaProjectionService
                startServerButton.setEnabled(true);
                stopServerButton.setEnabled(false);
                Intent serviceIntent = new Intent(MainActivity.this, MediaProjectionService.class);
                stopService(serviceIntent);
            }
        });



    }




    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_CODE_CAPTURE_PERM && resultCode == RESULT_OK) {
            Intent serviceIntent = new Intent(this, MediaProjectionService.class);
            serviceIntent.putExtra("resultCode", resultCode);
            serviceIntent.putExtra("data", data);

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                startForegroundService(serviceIntent);
            } else {
                startService(serviceIntent);
            }

        } else {
            Button startServer = findViewById(R.id.startserver);
            startServer.setEnabled(true);
        }
    }


}