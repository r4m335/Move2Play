using UnityEngine;

public class EnemyIndicator : MonoBehaviour
{
    public SpriteRenderer spriteRenderer;

    [Header("Flash")]
    public bool flashing = false;

    public float flashSpeed = 6f;
    public float minAlpha = 0.2f;
    public float maxAlpha = 1f;

    void Start()
    {
        if (spriteRenderer == null)
            spriteRenderer = GetComponent<SpriteRenderer>();
    }

    void Update()
    {
        if (!flashing)
        {
            Color normal = spriteRenderer.color;
            normal.a = 1f;
            spriteRenderer.color = normal;
            return;
        }

        float t = Mathf.PingPong(Time.time * flashSpeed, 1f);

        Color c = spriteRenderer.color;
        c.a = Mathf.Lerp(minAlpha, maxAlpha, t);

        spriteRenderer.color = c;
    }
}