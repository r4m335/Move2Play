using UnityEngine;

public class EnemySpawner : MonoBehaviour
{
    [Header("Prefabs")]
    public GameObject[] enemyPrefabs;

    [Header("Spawn Settings")]
    public int count = 20;                 // initial enemies
    public LayerMask groundLayer;
    public LayerMask waterLayer;
    public LayerMask enemyLayer;           // ★ layer(s) assigned to enemy GameObjects
    public float minDistance = 3f;
    public float maxSlopeAngle = 30f;

    [Header("Wave Spawning")]
    public float spawnInterval = 30f;
    public int enemiesPerWave = 3;

    private float timer;
    private UIController uiController;

    private void Start()
    {
        uiController = FindObjectOfType<UIController>();
        Spawn();                    // initial batch
        timer = spawnInterval;
    }

    private void Update()
    {
        // Stop spawning if the game has ended (death or victory)
        if (uiController != null && uiController.gameEnded)
            return;

        timer -= Time.deltaTime;
        if (timer <= 0f)
        {
            SpawnWave(enemiesPerWave);
            timer = spawnInterval;
        }
    }

    [ContextMenu("Spawn Enemies")]
    private void Spawn()
    {
        SpawnWave(count);
    }

    private void SpawnWave(int amount)
    {
        Terrain terrain = Terrain.activeTerrain;
        if (terrain == null)
        {
            Debug.LogError("No active terrain found!");
            return;
        }

        Vector3 terrainPos = terrain.transform.position;
        Vector3 terrainSize = terrain.terrainData.size;

        int spawned = 0;
        int attempts = 0;

        while (spawned < amount && attempts < amount * 10)
        {
            attempts++;

            float rx = Random.Range(0, terrainSize.x);
            float rz = Random.Range(0, terrainSize.z);
            Vector3 rayStart = new Vector3(terrainPos.x + rx, terrainPos.y + 200f, terrainPos.z + rz);

            if (Physics.Raycast(rayStart, Vector3.down, out RaycastHit hit, 300f, groundLayer))
            {
                // Reject steep slopes
                if (Vector3.Angle(hit.normal, Vector3.up) > maxSlopeAngle)
                    continue;

                Vector3 spawnPoint = hit.point;

                // Reject water
                if (((1 << hit.collider.gameObject.layer) & waterLayer) != 0)
                    continue;

                // Quick proximity check using dedicated enemy layer
                if (IsTooClose(spawnPoint))
                    continue;

                Quaternion rot = Quaternion.FromToRotation(Vector3.up, hit.normal) *
                                 Quaternion.Euler(0, Random.Range(0, 360), 0);

                GameObject prefab = enemyPrefabs[Random.Range(0, enemyPrefabs.Length)];
                Instantiate(prefab, spawnPoint, rot, transform);
                spawned++;
            }
        }

        Debug.Log($"Spawned {spawned} enemies (attempts: {attempts})");
    }

    // Only checks colliders on the enemyLayer – avoids terrain, trees, etc.
    bool IsTooClose(Vector3 pos)
    {
        Collider[] nearby = Physics.OverlapSphere(pos, minDistance, enemyLayer);
        foreach (Collider c in nearby)
        {
            if (c.CompareTag("Enemy"))   // extra safety
                return true;
        }
        return false;
    }

    void OnDrawGizmosSelected()
    {
        Terrain terrain = Terrain.activeTerrain;
        if (terrain != null)
        {
            Gizmos.color = Color.green;
            Vector3 size = terrain.terrainData.size;
            Gizmos.DrawWireCube(terrain.transform.position + size * 0.5f, size);
        }
    }
}