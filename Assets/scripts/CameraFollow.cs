using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    public Transform target;
    public Vector3 offset = new Vector3(0, 2, -4);
    public float smoothSpeed = 10f;
    public float fixedPitch = 15f;
    public float minDistance = 1.5f;
    public float maxDistance = 4f;
    
    [Tooltip("Layers that block the camera (walls, ground, etc.)")]
    public LayerMask collisionMask = ~0;          // Default: all layers
    
    [Tooltip("Layer used for enemies (will be ignored by camera collision)")]
    public LayerMask enemyLayerMask = 1 << 6;     // Example: layer 6 = "Enemy"

    private float currentDistance;
    private float targetDistance;
    private float distanceVelocity;
    private Vector3 currentVelocity;

    // Combined mask that excludes the enemy layer
    private LayerMask finalMask;

    void Start()
    {
        Cursor.lockState = CursorLockMode.None;
        Cursor.visible = true;

        currentDistance = maxDistance;
        targetDistance = maxDistance;
        
        // Exclude enemy layer from collision mask
        finalMask = collisionMask & ~enemyLayerMask;
    }

    void LateUpdate()
    {
        if (target == null) return;

        Vector3 playerForward = target.forward;
        playerForward.y = 0;
        playerForward.Normalize();

        float yaw = Mathf.Atan2(playerForward.x, playerForward.z) * Mathf.Rad2Deg;
        float pitch = fixedPitch;

        Quaternion rotation = Quaternion.Euler(pitch, yaw, 0);
        Vector3 desiredOffset = rotation * offset;
        float desiredDistance = desiredOffset.magnitude;

        Vector3 focusPoint = target.position + Vector3.up * 1.5f;
        Vector3 desiredDirection = desiredOffset.normalized;

        RaycastHit hit;

        // Use finalMask that ignores enemies
        if (Physics.Raycast(focusPoint, desiredDirection, out hit, desiredDistance, finalMask))
        {
            targetDistance = Mathf.Clamp(hit.distance - 0.2f, minDistance, maxDistance);
        }
        else
        {
            targetDistance = desiredDistance;
        }

        currentDistance = Mathf.SmoothDamp(currentDistance, targetDistance, ref distanceVelocity, 0.1f);
        Vector3 desiredPosition = focusPoint + desiredDirection * currentDistance;

        float actualSmoothSpeed = smoothSpeed;
        if (targetDistance < desiredDistance - 0.1f)
        {
            actualSmoothSpeed = smoothSpeed * 1.5f;
        }

        transform.position = Vector3.SmoothDamp(
            transform.position,
            desiredPosition,
            ref currentVelocity,
            1f / actualSmoothSpeed,
            Mathf.Infinity,
            Time.deltaTime
        );

        Quaternion targetRotation = Quaternion.LookRotation(focusPoint - transform.position);
        transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, smoothSpeed * Time.deltaTime);
    }
}