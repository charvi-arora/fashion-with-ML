# ============================================================
# outfit_model.py  —  OOP Classes: User and Outfit
# ============================================================

class User:
    VALID_BODY_TYPES  = ['pear','apple','rectangle','hourglass']
    VALID_SKIN_TONES  = ['warm','cool','neutral']
    VALID_OCCASIONS   = ['casual','party','formal']
    VALID_OUTFIT_TYPES= ['kurta','saree','lehenga','dress','formal_set','top_jeans',
                         'co_ord_set','jumpsuit','anarkali','palazzo_set',
                         'shirt_trouser','maxi_dress']
    VALID_SEASONS     = ['summer','winter','spring','autumn','all']

    def __init__(self, name, body_type, skin_tone, occasion,
                 outfit_type='dress', season='all'):
        self.name        = name
        self.body_type   = self._v('body_type',   body_type,   self.VALID_BODY_TYPES)
        self.skin_tone   = self._v('skin_tone',   skin_tone,   self.VALID_SKIN_TONES)
        self.occasion    = self._v('occasion',    occasion,    self.VALID_OCCASIONS)
        self.outfit_type = self._v('outfit_type', outfit_type, self.VALID_OUTFIT_TYPES)
        self.season      = self._v('season',      season,      self.VALID_SEASONS)

    def _v(self, field, value, valid):
        v = value.strip().lower()
        if v not in valid:
            raise ValueError(f"Invalid {field}: '{v}'. Choose from: {valid}")
        return v

    def as_feature_dict(self):
        return {'body_type':self.body_type,'skin_tone':self.skin_tone,
                'occasion':self.occasion,'outfit_type':self.outfit_type,'season':self.season}

    def __repr__(self):
        return f"User({self.name!r}|{self.body_type}|{self.skin_tone}|{self.occasion})"


class Outfit:
    def __init__(self, outfit_id, product_name, brand, outfit_type, color,
                 season, occasion, body_type, skin_tone, style_tag,
                 price_inr=0, rating=0.0, num_reviews=0):
        self.outfit_id    = outfit_id
        self.product_name = product_name
        self.brand        = brand
        self.outfit_type  = outfit_type
        self.color        = color
        self.season       = season
        self.occasion     = occasion
        self.body_type    = body_type
        self.skin_tone    = skin_tone
        self.style_tag    = style_tag
        self.price_inr    = price_inr
        self.rating       = rating
        self.num_reviews  = num_reviews

    def summary(self):
        return (f"[{self.outfit_id:>4}] {self.product_name:<45} "
                f"| {self.brand:<20} | ₹{self.price_inr:>6,} | ★{self.rating}")

    def __repr__(self):
        return f"Outfit(id={self.outfit_id}, name={self.product_name!r})"
