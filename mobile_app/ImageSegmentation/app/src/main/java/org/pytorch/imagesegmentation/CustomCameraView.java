package org.pytorch.imagesegmentation;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

public class CustomCameraView extends View {
    private Bitmap cameraFrame;
    private Bitmap overlayBitmap;
    private int targetWidth = 1000;
    private int targetHeight = 800;

    public CustomCameraView(Context context) {
        super(context);
    }

    public CustomCameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public CustomCameraView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public void setCameraFrame(Bitmap frame, boolean fromCamera) {

        if (fromCamera){
            this.targetHeight = 1080;
            this.targetWidth = 1080;
        }
        this.cameraFrame = Bitmap.createScaledBitmap(frame, targetWidth, targetHeight, true);
        invalidate(); // Trigger a redraw
    }

    public void setSegmentationMask(Bitmap overlay) {
        this.overlayBitmap = Bitmap.createScaledBitmap(overlay, targetWidth, targetHeight, true);
        invalidate(); // Trigger a redraw
    }
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int centerX = (getWidth() - targetWidth) / 2;
        int centerY = (getHeight() - targetHeight) / 2;

        // Draw the camera frame
        if (cameraFrame != null) {
            canvas.drawBitmap(cameraFrame, centerX, centerY, null);
        }

        // Draw the overlay with reduced opacity
        if (overlayBitmap != null) {
            Paint paint = new Paint();
            paint.setAlpha(50); // Adjust alpha value as needed (0-255), 128 is 50% transparency
            canvas.drawBitmap(overlayBitmap, centerX, centerY, paint);
        }
    }

}
