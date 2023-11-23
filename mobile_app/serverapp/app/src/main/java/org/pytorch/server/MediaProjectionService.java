package org.pytorch.server;

import android.app.Activity;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.util.DisplayMetrics;

import androidx.annotation.Nullable;
import androidx.core.app.NotificationCompat;

import org.pytorch.imagesegmentation.R;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;

public class MediaProjectionService extends Service {

    private static final int NOTIFICATION_ID = 1;
    private static final String CHANNEL_ID = "MediaProjectionChannel";

    private MediaProjectionManager projectionManager;

    private ServerSocket serverSocket;
    private static final int SERVER_PORT = 8080;
    private Socket clientSocket;
    private HandlerThread handlerThread;
    private Handler backgroundHandler;
    private volatile boolean isServerRunning = true; // Flag to control server thread

    private MediaProjection mediaProjection;
    private final byte[] FRAME_DELIMITER = "end".getBytes();

    private VirtualDisplay virtualDisplay;
    private ImageReader imageReader;
    private int screenDensity, width, height;
    private static final int desiredFrameRate = 5;

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            Notification notification = buildNotification(); // Your method to build a notification
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
        } else {
            startForeground(NOTIFICATION_ID, buildNotification());
        }    }

    private void startServer() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    serverSocket = new ServerSocket(SERVER_PORT);
                    while (isServerRunning) {
                        clientSocket = serverSocket.accept();
                        // Client connected
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }

    private void sendData(byte[] data) {
        try {
            if(clientSocket != null && !clientSocket.isClosed()) {
                OutputStream outputStream = clientSocket.getOutputStream();
                outputStream.write(data);
                outputStream.write(FRAME_DELIMITER);
                outputStream.flush();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        int resultCode = intent.getIntExtra("resultCode", Activity.RESULT_CANCELED);
        Intent projectionData = intent.getParcelableExtra("data");

        projectionManager = (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);
        mediaProjection = projectionManager.getMediaProjection(resultCode, projectionData);

        setupScreenCapture(); // Set up screen capture parameters and start capturing
        startScreenCapture(); // Start screen capture

        startServer();
        return START_STICKY;
    }

    private Bitmap centerCropToSquare(Bitmap bitmap) {
        // Keeping the full width and 75% of the height
        int width = bitmap.getWidth();
        int height = (int) (bitmap.getHeight() * 0.75);

        // Calculating the top coordinate for the crop to center the height
        int x = 0; // Keep the full width, no cropping from sides
        int y = (bitmap.getHeight() - height) / 2; // Cropping top and bottom to center

        // Creating the cropped bitmap
        Bitmap croppedBitmap = Bitmap.createBitmap(bitmap, x, y, width, height);

        return croppedBitmap;
    }
    private void setupScreenCapture() {
        // Initialize screen capture parameters like screenDensity, width, and height
        // This can be done similar to how it's done in MainActivity
        DisplayMetrics metrics = getResources().getDisplayMetrics();
        screenDensity = metrics.densityDpi;
        width = metrics.widthPixels;
        height = metrics.heightPixels;

        // Setup ImageReader
        imageReader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2);
        handlerThread = new HandlerThread("ScreenCaptureHandler");
        handlerThread.start();
        backgroundHandler = new Handler(handlerThread.getLooper());

        imageReader.setOnImageAvailableListener(new ImageReader.OnImageAvailableListener() {
            @Override
            public void onImageAvailable(ImageReader reader) {
                while (isServerRunning) {
                    try (Image img = reader.acquireLatestImage()) {
                        if (img != null) {
                            Image.Plane[] planes = img.getPlanes();
                            if (planes[0].getBuffer() == null) {
                                continue; // Skip if buffer is not available
                            }

                            ByteBuffer buffer = planes[0].getBuffer();
                            int pixelStride = planes[0].getPixelStride();
                            int rowStride = planes[0].getRowStride();
                            int rowPadding = rowStride - pixelStride * width;

                            Bitmap bitmap = Bitmap.createBitmap(width + rowPadding / pixelStride, height, Bitmap.Config.ARGB_8888);
                            bitmap.copyPixelsFromBuffer(buffer);
                            bitmap = centerCropToSquare(bitmap);
                            bitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true);
                            // Compress the Bitmap to JPEG
                            try (ByteArrayOutputStream stream = new ByteArrayOutputStream()) {
                                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
                                byte[] data = stream.toByteArray();
                                sendData(data); // Send data to the client
                            }

                            // Control the frame rate (optional)
                            Thread.sleep(1000 / desiredFrameRate);
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        }, backgroundHandler);
    }

    private void startScreenCapture() {
        virtualDisplay = mediaProjection.createVirtualDisplay("ScreenCapture",
                width, height, screenDensity,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader.getSurface(), null, null);
    }



    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID,
                    "Media Projection",
                    NotificationManager.IMPORTANCE_HIGH
            );
            channel.setDescription("Media Projection Service");
            channel.enableLights(true);
            channel.setLightColor(Color.RED);
            channel.enableVibration(false);

            NotificationManager manager = getSystemService(NotificationManager.class);
            manager.createNotificationChannel(channel);
        }
    }

    private Notification buildNotification() {
        Intent notificationIntent = new Intent(this, MainActivity.class);

        PendingIntent pendingIntent;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            // For Android 12 (API level 31) and above, specify the PendingIntent mutability
            pendingIntent = PendingIntent.getActivity(
                    this,
                    0,
                    notificationIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
            );
        } else {
            // For older versions, use the existing method
            pendingIntent = PendingIntent.getActivity(
                    this,
                    0,
                    notificationIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT
            );
        }

        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("Media Projection Service")
                .setContentText("Recording screen")
                .setSmallIcon(R.mipmap.ic_launcher)
                .setContentIntent(pendingIntent)
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setCategory(NotificationCompat.CATEGORY_SERVICE)
                .build();
    }

    private void stopServer() {
        isServerRunning = false; // Signal the server thread to stop
        try {
            if (clientSocket != null && !clientSocket.isClosed()) {
                clientSocket.close(); // Close client socket
            }
            if (serverSocket != null && !serverSocket.isClosed()) {
                serverSocket.close(); // Close server socket
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        stopServer();
        isServerRunning = false; // Signal the server thread to stop
        if (virtualDisplay != null) {
            virtualDisplay.release(); // Release the virtual display
        }
        if (mediaProjection != null) {
            mediaProjection.stop(); // Stop the media projection
        }
        if (serverSocket != null && !serverSocket.isClosed()) {
            try {
                serverSocket.close(); // Close the server socket
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
