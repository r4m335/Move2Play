using UnityEngine;

public class MinimapCameraFollow : MonoBehaviour
{
    public Transform player;            // Drag the player here
    public float heightAbovePlayer = 50f;

    private void Start()
    {
        if (player == null)
            player = GameObject.FindGameObjectWithTag("Player").transform;

        // Point straight down
        transform.rotation = Quaternion.Euler(90f, 0f, 0f);
    }

    private void LateUpdate()
    {
        if (player == null) return;

        // Stay directly above the player
        Vector3 pos = player.position;
        pos.y += heightAbovePlayer;
        transform.position = pos;

        // Keep the camera pointing straight down (north-up)
        transform.rotation = Quaternion.Euler(90f, 0f, 0f);
    }
}