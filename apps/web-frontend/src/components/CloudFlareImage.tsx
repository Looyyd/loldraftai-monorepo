// This component is used to load images from the CloudFlare CDN.
// Used to avoid high egress costs of Vercel.
const CloudFlareImage = (
  props: Omit<React.ImgHTMLAttributes<HTMLImageElement>, "src"> & {
    src: string;
  }
) => {
  // Construct the full URL directly
  const fullSrc = `https://media.loldraftai.com${props.src}`;

  return (
    <img
      {...props}
      src={fullSrc}
      alt={props.alt || "Image"} // Maintain the fallback alt text for accessibility
      loading="lazy" // Keep lazy loading behavior
    />
  );
};

export default CloudFlareImage;
